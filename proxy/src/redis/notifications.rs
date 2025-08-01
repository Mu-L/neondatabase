use std::convert::Infallible;
use std::sync::Arc;

use futures::StreamExt;
use redis::aio::PubSub;
use serde::Deserialize;
use tokio_util::sync::CancellationToken;

use super::connection_with_credentials_provider::ConnectionWithCredentialsProvider;
use crate::cache::project_info::ProjectInfoCache;
use crate::intern::{AccountIdInt, EndpointIdInt, ProjectIdInt, RoleNameInt};
use crate::metrics::{Metrics, RedisErrors, RedisEventsCount};
use crate::util::deserialize_json_string;

const CPLANE_CHANNEL_NAME: &str = "neondb-proxy-ws-updates";
const RECONNECT_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(20);
const INVALIDATION_LAG: std::time::Duration = std::time::Duration::from_secs(20);

async fn try_connect(client: &ConnectionWithCredentialsProvider) -> anyhow::Result<PubSub> {
    let mut conn = client.get_async_pubsub().await?;
    tracing::info!("subscribing to a channel `{CPLANE_CHANNEL_NAME}`");
    conn.subscribe(CPLANE_CHANNEL_NAME).await?;
    Ok(conn)
}

#[derive(Debug, Deserialize)]
struct NotificationHeader<'a> {
    topic: &'a str,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(tag = "topic", content = "data")]
enum Notification {
    #[serde(
        rename = "/account_settings_update",
        alias = "/allowed_vpc_endpoints_updated_for_org",
        deserialize_with = "deserialize_json_string"
    )]
    AccountSettingsUpdate(InvalidateAccount),

    #[serde(
        rename = "/endpoint_settings_update",
        deserialize_with = "deserialize_json_string"
    )]
    EndpointSettingsUpdate(InvalidateEndpoint),

    #[serde(
        rename = "/project_settings_update",
        alias = "/allowed_ips_updated",
        alias = "/block_public_or_vpc_access_updated",
        alias = "/allowed_vpc_endpoints_updated_for_projects",
        deserialize_with = "deserialize_json_string"
    )]
    ProjectSettingsUpdate(InvalidateProject),

    #[serde(
        rename = "/role_setting_update",
        alias = "/password_updated",
        deserialize_with = "deserialize_json_string"
    )]
    RoleSettingUpdate(InvalidateRole),

    #[serde(
        other,
        deserialize_with = "deserialize_unknown_topic",
        skip_serializing
    )]
    UnknownTopic,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
enum InvalidateEndpoint {
    EndpointId(EndpointIdInt),
    EndpointIds(Vec<EndpointIdInt>),
}
impl std::ops::Deref for InvalidateEndpoint {
    type Target = [EndpointIdInt];
    fn deref(&self) -> &Self::Target {
        match self {
            Self::EndpointId(id) => std::slice::from_ref(id),
            Self::EndpointIds(ids) => ids,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
enum InvalidateProject {
    ProjectId(ProjectIdInt),
    ProjectIds(Vec<ProjectIdInt>),
}
impl std::ops::Deref for InvalidateProject {
    type Target = [ProjectIdInt];
    fn deref(&self) -> &Self::Target {
        match self {
            Self::ProjectId(id) => std::slice::from_ref(id),
            Self::ProjectIds(ids) => ids,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
enum InvalidateAccount {
    AccountId(AccountIdInt),
    AccountIds(Vec<AccountIdInt>),
}
impl std::ops::Deref for InvalidateAccount {
    type Target = [AccountIdInt];
    fn deref(&self) -> &Self::Target {
        match self {
            Self::AccountId(id) => std::slice::from_ref(id),
            Self::AccountIds(ids) => ids,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
struct InvalidateRole {
    project_id: ProjectIdInt,
    role_name: RoleNameInt,
}

// https://github.com/serde-rs/serde/issues/1714
fn deserialize_unknown_topic<'de, D>(deserializer: D) -> Result<(), D::Error>
where
    D: serde::Deserializer<'de>,
{
    deserializer.deserialize_any(serde::de::IgnoredAny)?;
    Ok(())
}

struct MessageHandler<C: Send + Sync + 'static> {
    cache: Arc<C>,
}

impl<C: Send + Sync + 'static> Clone for MessageHandler<C> {
    fn clone(&self) -> Self {
        Self {
            cache: self.cache.clone(),
        }
    }
}

impl MessageHandler<ProjectInfoCache> {
    pub(crate) fn new(cache: Arc<ProjectInfoCache>) -> Self {
        Self { cache }
    }

    #[tracing::instrument(skip(self, msg), fields(session_id = tracing::field::Empty))]
    async fn handle_message(&self, msg: redis::Msg) -> anyhow::Result<()> {
        let payload: String = msg.get_payload()?;
        tracing::debug!(?payload, "received a message payload");

        let msg: Notification = match serde_json::from_str(&payload) {
            Ok(Notification::UnknownTopic) => {
                match serde_json::from_str::<NotificationHeader>(&payload) {
                    // don't update the metric for redis errors if it's just a topic we don't know about.
                    Ok(header) => tracing::warn!(topic = header.topic, "unknown topic"),
                    Err(e) => {
                        Metrics::get().proxy.redis_errors_total.inc(RedisErrors {
                            channel: msg.get_channel_name(),
                        });
                        tracing::error!("broken message: {e}");
                    }
                }
                return Ok(());
            }
            Ok(msg) => msg,
            Err(e) => {
                Metrics::get().proxy.redis_errors_total.inc(RedisErrors {
                    channel: msg.get_channel_name(),
                });
                match serde_json::from_str::<NotificationHeader>(&payload) {
                    Ok(header) => tracing::error!(topic = header.topic, "broken message: {e}"),
                    Err(_) => tracing::error!("broken message: {e}"),
                }
                return Ok(());
            }
        };

        tracing::debug!(?msg, "received a message");
        match msg {
            Notification::RoleSettingUpdate { .. }
            | Notification::EndpointSettingsUpdate { .. }
            | Notification::ProjectSettingsUpdate { .. }
            | Notification::AccountSettingsUpdate { .. } => {
                invalidate_cache(self.cache.clone(), msg.clone());

                let m = &Metrics::get().proxy.redis_events_count;
                match msg {
                    Notification::RoleSettingUpdate { .. } => {
                        m.inc(RedisEventsCount::InvalidateRole);
                    }
                    Notification::EndpointSettingsUpdate { .. } => {
                        m.inc(RedisEventsCount::InvalidateEndpoint);
                    }
                    Notification::ProjectSettingsUpdate { .. } => {
                        m.inc(RedisEventsCount::InvalidateProject);
                    }
                    Notification::AccountSettingsUpdate { .. } => {
                        m.inc(RedisEventsCount::InvalidateOrg);
                    }
                    Notification::UnknownTopic => {}
                }

                // TODO: add additional metrics for the other event types.

                // It might happen that the invalid entry is on the way to be cached.
                // To make sure that the entry is invalidated, let's repeat the invalidation in INVALIDATION_LAG seconds.
                // TODO: include the version (or the timestamp) in the message and invalidate only if the entry is cached before the message.
                let cache = self.cache.clone();
                tokio::spawn(async move {
                    tokio::time::sleep(INVALIDATION_LAG).await;
                    invalidate_cache(cache, msg);
                });
            }

            Notification::UnknownTopic => unreachable!(),
        }

        Ok(())
    }
}

fn invalidate_cache(cache: Arc<ProjectInfoCache>, msg: Notification) {
    match msg {
        Notification::EndpointSettingsUpdate(ids) => ids
            .iter()
            .for_each(|&id| cache.invalidate_endpoint_access(id)),

        Notification::AccountSettingsUpdate(ids) => ids
            .iter()
            .for_each(|&id| cache.invalidate_endpoint_access_for_org(id)),

        Notification::ProjectSettingsUpdate(ids) => ids
            .iter()
            .for_each(|&id| cache.invalidate_endpoint_access_for_project(id)),

        Notification::RoleSettingUpdate(InvalidateRole {
            project_id,
            role_name,
        }) => cache.invalidate_role_secret_for_project(project_id, role_name),

        Notification::UnknownTopic => unreachable!(),
    }
}

async fn handle_messages(
    handler: MessageHandler<ProjectInfoCache>,
    redis: ConnectionWithCredentialsProvider,
    cancellation_token: CancellationToken,
) -> anyhow::Result<()> {
    loop {
        if cancellation_token.is_cancelled() {
            return Ok(());
        }
        let mut conn = match try_connect(&redis).await {
            Ok(conn) => conn,
            Err(e) => {
                tracing::error!(
                    "failed to connect to redis: {e}, will try to reconnect in {RECONNECT_TIMEOUT:#?}"
                );
                tokio::time::sleep(RECONNECT_TIMEOUT).await;
                continue;
            }
        };
        let mut stream = conn.on_message();
        while let Some(msg) = stream.next().await {
            match handler.handle_message(msg).await {
                Ok(()) => {}
                Err(e) => {
                    tracing::error!("failed to handle message: {e}, will try to reconnect");
                    break;
                }
            }
            if cancellation_token.is_cancelled() {
                return Ok(());
            }
        }
    }
}

/// Handle console's invalidation messages.
#[tracing::instrument(name = "redis_notifications", skip_all)]
pub async fn task_main(
    redis: ConnectionWithCredentialsProvider,
    cache: Arc<ProjectInfoCache>,
) -> anyhow::Result<Infallible> {
    let handler = MessageHandler::new(cache);
    // 6h - 1m.
    // There will be 1 minute overlap between two tasks. But at least we can be sure that no message is lost.
    let mut interval = tokio::time::interval(std::time::Duration::from_secs(6 * 60 * 60 - 60));
    loop {
        let cancellation_token = CancellationToken::new();
        interval.tick().await;

        tokio::spawn(handle_messages(
            handler.clone(),
            redis.clone(),
            cancellation_token.clone(),
        ));
        tokio::spawn(async move {
            tokio::time::sleep(std::time::Duration::from_secs(6 * 60 * 60)).await; // 6h.
            cancellation_token.cancel();
        });
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::types::{ProjectId, RoleName};

    #[test]
    fn parse_allowed_ips() -> anyhow::Result<()> {
        let project_id: ProjectId = "new_project".into();
        let data = format!("{{\"project_id\": \"{project_id}\"}}");
        let text = json!({
            "type": "message",
            "topic": "/allowed_ips_updated",
            "data": data,
            "extre_fields": "something"
        })
        .to_string();

        let result: Notification = serde_json::from_str(&text)?;
        assert_eq!(
            result,
            Notification::ProjectSettingsUpdate(InvalidateProject::ProjectId((&project_id).into()))
        );

        Ok(())
    }

    #[test]
    fn parse_multiple_projects() -> anyhow::Result<()> {
        let project_id1: ProjectId = "new_project1".into();
        let project_id2: ProjectId = "new_project2".into();
        let data = format!("{{\"project_ids\": [\"{project_id1}\",\"{project_id2}\"]}}");
        let text = json!({
            "type": "message",
            "topic": "/allowed_vpc_endpoints_updated_for_projects",
            "data": data,
            "extre_fields": "something"
        })
        .to_string();

        let result: Notification = serde_json::from_str(&text)?;
        assert_eq!(
            result,
            Notification::ProjectSettingsUpdate(InvalidateProject::ProjectIds(vec![
                (&project_id1).into(),
                (&project_id2).into()
            ]))
        );

        Ok(())
    }

    #[test]
    fn parse_password_updated() -> anyhow::Result<()> {
        let project_id: ProjectId = "new_project".into();
        let role_name: RoleName = "new_role".into();
        let data = format!("{{\"project_id\": \"{project_id}\", \"role_name\": \"{role_name}\"}}");
        let text = json!({
            "type": "message",
            "topic": "/password_updated",
            "data": data,
            "extre_fields": "something"
        })
        .to_string();

        let result: Notification = serde_json::from_str(&text)?;
        assert_eq!(
            result,
            Notification::RoleSettingUpdate(InvalidateRole {
                project_id: (&project_id).into(),
                role_name: (&role_name).into(),
            })
        );

        Ok(())
    }

    #[test]
    fn parse_unknown_topic() -> anyhow::Result<()> {
        let with_data = json!({
            "type": "message",
            "topic": "/doesnotexist",
            "data": {
                "payload": "ignored"
            },
            "extra_fields": "something"
        })
        .to_string();
        let result: Notification = serde_json::from_str(&with_data)?;
        assert_eq!(result, Notification::UnknownTopic);

        let without_data = json!({
            "type": "message",
            "topic": "/doesnotexist",
            "extra_fields": "something"
        })
        .to_string();
        let result: Notification = serde_json::from_str(&without_data)?;
        assert_eq!(result, Notification::UnknownTopic);

        Ok(())
    }
}
