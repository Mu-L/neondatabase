use std::collections::{HashMap, HashSet};
use std::fmt::Display;
use std::net::IpAddr;
use std::str::FromStr;
use std::time::{Duration, Instant};

/// Request/response types for the storage controller
/// API (`/control/v1` prefix).  Implemented by the server
/// in [`storage_controller::http`]
use serde::{Deserialize, Serialize};
use utils::id::{NodeId, TenantId, TimelineId};
use utils::lsn::Lsn;

use crate::models::{PageserverUtilization, ShardParameters, TenantConfig, TimelineInfo};
use crate::shard::{ShardStripeSize, TenantShardId};

#[derive(Serialize, Deserialize, Debug)]
#[serde(deny_unknown_fields)]
pub struct TenantCreateRequest {
    pub new_tenant_id: TenantShardId,
    #[serde(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub generation: Option<u32>,

    // If omitted, create a single shard with TenantShardId::unsharded()
    #[serde(default)]
    #[serde(skip_serializing_if = "ShardParameters::is_unsharded")]
    pub shard_parameters: ShardParameters,

    #[serde(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub placement_policy: Option<PlacementPolicy>,

    #[serde(flatten)]
    pub config: TenantConfig, // as we have a flattened field, we should reject all unknown fields in it
}

#[derive(Serialize, Deserialize)]
pub struct TenantCreateResponseShard {
    pub shard_id: TenantShardId,
    pub node_id: NodeId,
    pub generation: u32,
}

#[derive(Serialize, Deserialize)]
pub struct TenantCreateResponse {
    pub shards: Vec<TenantCreateResponseShard>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct NodeRegisterRequest {
    pub node_id: NodeId,

    pub listen_pg_addr: String,
    pub listen_pg_port: u16,
    pub listen_grpc_addr: Option<String>,
    pub listen_grpc_port: Option<u16>,

    pub listen_http_addr: String,
    pub listen_http_port: u16,
    pub listen_https_port: Option<u16>,

    pub availability_zone_id: AvailabilityZone,

    // Reachable IP address of the PS/SK registering, if known.
    // Hadron Cluster Coordiantor will update the DNS record of the registering node
    // with this IP address.
    pub node_ip_addr: Option<IpAddr>,
}

#[derive(Serialize, Deserialize)]
pub struct NodeConfigureRequest {
    pub node_id: NodeId,

    pub availability: Option<NodeAvailabilityWrapper>,
    pub scheduling: Option<NodeSchedulingPolicy>,
}

#[derive(Serialize, Deserialize)]
pub struct TenantPolicyRequest {
    pub placement: Option<PlacementPolicy>,
    pub scheduling: Option<ShardSchedulingPolicy>,
}

#[derive(Clone, Serialize, Deserialize, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub struct AvailabilityZone(pub String);

impl Display for AvailabilityZone {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[derive(Serialize, Deserialize)]
pub struct ShardsPreferredAzsRequest {
    #[serde(flatten)]
    pub preferred_az_ids: HashMap<TenantShardId, Option<AvailabilityZone>>,
}

#[derive(Serialize, Deserialize)]
pub struct ShardsPreferredAzsResponse {
    pub updated: Vec<TenantShardId>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct TenantLocateResponseShard {
    pub shard_id: TenantShardId,
    pub node_id: NodeId,

    pub listen_pg_addr: String,
    pub listen_pg_port: u16,
    pub listen_grpc_addr: Option<String>,
    pub listen_grpc_port: Option<u16>,

    pub listen_http_addr: String,
    pub listen_http_port: u16,
    pub listen_https_port: Option<u16>,
}

#[derive(Serialize, Deserialize)]
pub struct TenantLocateResponse {
    pub shards: Vec<TenantLocateResponseShard>,
    pub shard_params: ShardParameters,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct TenantDescribeResponse {
    pub tenant_id: TenantId,
    pub shards: Vec<TenantDescribeResponseShard>,
    pub stripe_size: ShardStripeSize,
    pub policy: PlacementPolicy,
    pub config: TenantConfig,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct TenantTimelineDescribeResponse {
    pub shards: Vec<TimelineInfo>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub image_consistent_lsn: Option<Lsn>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct NodeShardResponse {
    pub node_id: NodeId,
    pub shards: Vec<NodeShard>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct NodeShard {
    pub tenant_shard_id: TenantShardId,
    /// Whether the shard is observed secondary on a specific node. True = yes, False = no, None = not on this node.
    pub is_observed_secondary: Option<bool>,
    /// Whether the shard is intended to be a secondary on a specific node. True = yes, False = no, None = not on this node.
    pub is_intended_secondary: Option<bool>,
}

#[derive(Serialize, Deserialize)]
pub struct NodeDescribeResponse {
    pub id: NodeId,

    pub availability: NodeAvailabilityWrapper,
    pub scheduling: NodeSchedulingPolicy,

    pub availability_zone_id: String,

    pub listen_http_addr: String,
    pub listen_http_port: u16,
    pub listen_https_port: Option<u16>,

    pub listen_pg_addr: String,
    pub listen_pg_port: u16,
    pub listen_grpc_addr: Option<String>,
    pub listen_grpc_port: Option<u16>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct TenantDescribeResponseShard {
    pub tenant_shard_id: TenantShardId,

    pub node_attached: Option<NodeId>,
    pub node_secondary: Vec<NodeId>,

    pub last_error: String,

    /// A task is currently running to reconcile this tenant's intent state with the state on pageservers
    pub is_reconciling: bool,
    /// This shard failed in sending a compute notification to the cloud control plane, and a retry is pending.
    pub is_pending_compute_notification: bool,
    /// A shard split is currently underway
    pub is_splitting: bool,
    /// A timeline is being imported into this tenant
    pub is_importing: bool,

    pub scheduling_policy: ShardSchedulingPolicy,

    pub preferred_az_id: Option<String>,
}

/// Migration request for a given tenant shard to a given node.
///
/// Explicitly migrating a particular shard is a low level operation
/// TODO: higher level "Reschedule tenant" operation where the request
/// specifies some constraints, e.g. asking it to get off particular node(s)
#[derive(Serialize, Deserialize, Debug)]
pub struct TenantShardMigrateRequest {
    pub node_id: NodeId,

    /// Optionally, callers may specify the node they are migrating _from_, and the server will
    /// reject the request if the shard is no longer attached there: this enables writing safer
    /// clients that don't risk fighting with some other movement of the shard.
    #[serde(default)]
    pub origin_node_id: Option<NodeId>,

    #[serde(default)]
    pub migration_config: MigrationConfig,
}

#[derive(Serialize, Deserialize, Debug, PartialEq, Eq)]
pub struct MigrationConfig {
    /// If true, the migration will be executed even if it is to a location with a sub-optimal scheduling
    /// score: this is usually not what you want, and if you use this then you'll also need to set the
    /// tenant's scheduling policy to Essential or Pause to avoid the optimiser reverting your migration.
    ///
    /// Default: false
    #[serde(default)]
    pub override_scheduler: bool,

    /// If true, the migration will be done gracefully by creating a secondary location first and
    /// waiting for it to warm up before cutting over.  If false, if there is no existing secondary
    /// location at the destination, the tenant will be migrated immediately.  If the tenant's data
    /// can't be downloaded within [`Self::secondary_warmup_timeout`], then the migration will go
    /// ahead but run with a cold cache that can severely reduce performance until it warms up.
    ///
    /// When doing a graceful migration, the migration API returns as soon as it is started.
    ///
    /// Default: true
    #[serde(default = "default_prewarm")]
    pub prewarm: bool,

    /// For non-prewarm migrations which will immediately enter a cutover to the new node: how long to wait
    /// overall for secondary warmup before cutting over
    #[serde(default)]
    #[serde(with = "humantime_serde")]
    pub secondary_warmup_timeout: Option<Duration>,
    /// For non-prewarm migrations which will immediately enter a cutover to the new node: how long to wait
    /// within each secondary download poll call to pageserver.
    #[serde(default)]
    #[serde(with = "humantime_serde")]
    pub secondary_download_request_timeout: Option<Duration>,
}

fn default_prewarm() -> bool {
    true
}

impl Default for MigrationConfig {
    fn default() -> Self {
        Self {
            override_scheduler: false,
            prewarm: default_prewarm(),
            secondary_warmup_timeout: None,
            secondary_download_request_timeout: None,
        }
    }
}

#[derive(Serialize, Clone, Debug)]
#[serde(into = "NodeAvailabilityWrapper")]
pub enum NodeAvailability {
    // Normal, happy state
    Active(PageserverUtilization),
    // Node is warming up, but we expect it to become available soon. Covers
    // the time span between the re-attach response being composed on the storage controller
    // and the first successful heartbeat after the processing of the re-attach response
    // finishes on the pageserver.
    WarmingUp(Instant),
    // Offline: Tenants shouldn't try to attach here, but they may assume that their
    // secondary locations on this node still exist.  Newly added nodes are in this
    // state until we successfully contact them.
    Offline,
}

impl PartialEq for NodeAvailability {
    fn eq(&self, other: &Self) -> bool {
        use NodeAvailability::*;
        matches!(
            (self, other),
            (Active(_), Active(_)) | (Offline, Offline) | (WarmingUp(_), WarmingUp(_))
        )
    }
}

impl Eq for NodeAvailability {}

// This wrapper provides serde functionality and it should only be used to
// communicate with external callers which don't know or care about the
// utilisation score of the pageserver it is targeting.
#[derive(Serialize, Deserialize, Clone, Copy, Debug)]
pub enum NodeAvailabilityWrapper {
    Active,
    WarmingUp,
    Offline,
}

impl From<NodeAvailabilityWrapper> for NodeAvailability {
    fn from(val: NodeAvailabilityWrapper) -> Self {
        match val {
            // Assume the worst utilisation score to begin with. It will later be updated by
            // the heartbeats.
            NodeAvailabilityWrapper::Active => {
                NodeAvailability::Active(PageserverUtilization::full())
            }
            NodeAvailabilityWrapper::WarmingUp => NodeAvailability::WarmingUp(Instant::now()),
            NodeAvailabilityWrapper::Offline => NodeAvailability::Offline,
        }
    }
}

impl From<NodeAvailability> for NodeAvailabilityWrapper {
    fn from(val: NodeAvailability) -> Self {
        match val {
            NodeAvailability::Active(_) => NodeAvailabilityWrapper::Active,
            NodeAvailability::WarmingUp(_) => NodeAvailabilityWrapper::WarmingUp,
            NodeAvailability::Offline => NodeAvailabilityWrapper::Offline,
        }
    }
}

/// Scheduling policy enables us to selectively disable some automatic actions that the
/// controller performs on a tenant shard. This is only set to a non-default value by
/// human intervention, and it is reset to the default value (Active) when the tenant's
/// placement policy is modified away from Attached.
///
/// The typical use of a non-Active scheduling policy is one of:
/// - Pinnning a shard to a node (i.e. migrating it there & setting a non-Active scheduling policy)
/// - Working around a bug (e.g. if something is flapping and we need to stop it until the bug is fixed)
///
/// If you're not sure which policy to use to pin a shard to its current location, you probably
/// want Pause.
#[derive(Serialize, Deserialize, Clone, Copy, Eq, PartialEq, Debug)]
pub enum ShardSchedulingPolicy {
    // Normal mode: the tenant's scheduled locations may be updated at will, including
    // for non-essential optimization.
    Active,

    // Disable optimizations, but permit scheduling when necessary to fulfil the PlacementPolicy.
    // For example, this still permits a node's attachment location to change to a secondary in
    // response to a node failure, or to assign a new secondary if a node was removed.
    Essential,

    // No scheduling: leave the shard running wherever it currently is.  Even if the shard is
    // unavailable, it will not be rescheduled to another node.
    Pause,

    // No reconciling: we will make no location_conf API calls to pageservers at all.  If the
    // shard is unavailable, it stays that way.  If a node fails, this shard doesn't get failed over.
    Stop,
}

impl Default for ShardSchedulingPolicy {
    fn default() -> Self {
        Self::Active
    }
}

#[derive(Serialize, Deserialize, Clone, Copy, Eq, PartialEq, Debug)]
pub enum NodeLifecycle {
    Active,
    Deleted,
}

impl FromStr for NodeLifecycle {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "active" => Ok(Self::Active),
            "deleted" => Ok(Self::Deleted),
            _ => Err(anyhow::anyhow!("Unknown node lifecycle '{s}'")),
        }
    }
}

impl From<NodeLifecycle> for String {
    fn from(value: NodeLifecycle) -> String {
        use NodeLifecycle::*;
        match value {
            Active => "active",
            Deleted => "deleted",
        }
        .to_string()
    }
}

#[derive(Serialize, Deserialize, Clone, Copy, Eq, PartialEq, Debug)]
pub enum NodeSchedulingPolicy {
    Active,
    Filling,
    Pause,
    PauseForRestart,
    Draining,
    Deleting,
}

impl FromStr for NodeSchedulingPolicy {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "active" => Ok(Self::Active),
            "filling" => Ok(Self::Filling),
            "pause" => Ok(Self::Pause),
            "pause_for_restart" => Ok(Self::PauseForRestart),
            "draining" => Ok(Self::Draining),
            "deleting" => Ok(Self::Deleting),
            _ => Err(anyhow::anyhow!("Unknown scheduling state '{s}'")),
        }
    }
}

impl From<NodeSchedulingPolicy> for String {
    fn from(value: NodeSchedulingPolicy) -> String {
        use NodeSchedulingPolicy::*;
        match value {
            Active => "active",
            Filling => "filling",
            Pause => "pause",
            PauseForRestart => "pause_for_restart",
            Draining => "draining",
            Deleting => "deleting",
        }
        .to_string()
    }
}

#[derive(Serialize, Deserialize, Clone, Copy, Eq, PartialEq, Debug)]
pub enum SkSchedulingPolicy {
    Active,
    Activating,
    Pause,
    Decomissioned,
}

impl FromStr for SkSchedulingPolicy {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(match s {
            "active" => Self::Active,
            "activating" => Self::Activating,
            "pause" => Self::Pause,
            "decomissioned" => Self::Decomissioned,
            _ => {
                return Err(anyhow::anyhow!(
                    "Unknown scheduling policy '{s}', try active,pause,decomissioned"
                ));
            }
        })
    }
}

impl From<SkSchedulingPolicy> for String {
    fn from(value: SkSchedulingPolicy) -> String {
        use SkSchedulingPolicy::*;
        match value {
            Active => "active",
            Activating => "activating",
            Pause => "pause",
            Decomissioned => "decomissioned",
        }
        .to_string()
    }
}

/// Controls how tenant shards are mapped to locations on pageservers, e.g. whether
/// to create secondary locations.
#[derive(Clone, Serialize, Deserialize, Debug, PartialEq, Eq)]
pub enum PlacementPolicy {
    /// Normal live state: one attached pageserver and zero or more secondaries.
    Attached(usize),
    /// Create one secondary mode locations. This is useful when onboarding
    /// a tenant, or for an idle tenant that we might want to bring online quickly.
    Secondary,

    /// Do not attach to any pageservers.  This is appropriate for tenants that
    /// have been idle for a long time, where we do not mind some delay in making
    /// them available in future.
    Detached,
}

impl PlacementPolicy {
    pub fn want_secondaries(&self) -> usize {
        match self {
            PlacementPolicy::Attached(secondary_count) => *secondary_count,
            PlacementPolicy::Secondary => 1,
            PlacementPolicy::Detached => 0,
        }
    }
}

#[derive(Serialize, Deserialize, Debug)]
pub struct TenantShardMigrateResponse {}

/// Metadata health record posted from scrubber.
#[derive(Serialize, Deserialize, Debug)]
pub struct MetadataHealthRecord {
    pub tenant_shard_id: TenantShardId,
    pub healthy: bool,
    pub last_scrubbed_at: chrono::DateTime<chrono::Utc>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct MetadataHealthUpdateRequest {
    pub healthy_tenant_shards: HashSet<TenantShardId>,
    pub unhealthy_tenant_shards: HashSet<TenantShardId>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct MetadataHealthUpdateResponse {}

#[derive(Serialize, Deserialize, Debug)]
pub struct MetadataHealthListUnhealthyResponse {
    pub unhealthy_tenant_shards: Vec<TenantShardId>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct MetadataHealthListOutdatedRequest {
    #[serde(with = "humantime_serde")]
    pub not_scrubbed_for: Duration,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct MetadataHealthListOutdatedResponse {
    pub health_records: Vec<MetadataHealthRecord>,
}

/// Publicly exposed safekeeper description
#[derive(Serialize, Deserialize, Clone)]
pub struct SafekeeperDescribeResponse {
    pub id: NodeId,
    pub region_id: String,
    /// 1 is special, it means just created (not currently posted to storcon).
    /// Zero or negative is not really expected.
    /// Otherwise the number from `release-$(number_of_commits_on_branch)` tag.
    pub version: i64,
    pub host: String,
    pub port: i32,
    pub http_port: i32,
    pub https_port: Option<i32>,
    pub availability_zone_id: String,
    pub scheduling_policy: SkSchedulingPolicy,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct TimelineSafekeeperPeer {
    pub node_id: NodeId,
    pub listen_http_addr: String,
    pub http_port: i32,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct SCSafekeeperTimeline {
    // SC does not know the tenant id.
    pub timeline_id: TimelineId,
    pub peers: Vec<NodeId>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct SCSafekeeperTimelinesResponse {
    pub timelines: Vec<SCSafekeeperTimeline>,
    pub safekeeper_peers: Vec<TimelineSafekeeperPeer>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct SafekeeperTimeline {
    pub tenant_id: TenantId,
    pub timeline_id: TimelineId,
    pub peers: Vec<NodeId>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct SafekeeperTimelinesResponse {
    pub timelines: Vec<SafekeeperTimeline>,
    pub safekeeper_peers: Vec<TimelineSafekeeperPeer>,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct SafekeeperSchedulingPolicyRequest {
    pub scheduling_policy: SkSchedulingPolicy,
}

/// Import request for safekeeper timelines.
#[derive(Serialize, Deserialize, Clone)]
pub struct TimelineImportRequest {
    pub tenant_id: TenantId,
    pub timeline_id: TimelineId,
    pub start_lsn: Lsn,
    pub sk_set: Vec<NodeId>,
    pub force_upsert: bool,
}

#[derive(serde::Serialize, serde::Deserialize, Clone)]
pub struct TimelineSafekeeperMigrateRequest {
    pub new_sk_set: Vec<NodeId>,
}

#[cfg(test)]
mod test {
    use serde_json;

    use super::*;

    /// Check stability of PlacementPolicy's serialization
    #[test]
    fn placement_policy_encoding() -> anyhow::Result<()> {
        let v = PlacementPolicy::Attached(1);
        let encoded = serde_json::to_string(&v)?;
        assert_eq!(encoded, "{\"Attached\":1}");
        assert_eq!(serde_json::from_str::<PlacementPolicy>(&encoded)?, v);

        let v = PlacementPolicy::Detached;
        let encoded = serde_json::to_string(&v)?;
        assert_eq!(encoded, "\"Detached\"");
        assert_eq!(serde_json::from_str::<PlacementPolicy>(&encoded)?, v);
        Ok(())
    }

    #[test]
    fn test_reject_unknown_field() {
        let id = TenantId::generate();
        let create_request = serde_json::json!({
            "new_tenant_id": id.to_string(),
            "unknown_field": "unknown_value".to_string(),
        });
        let err = serde_json::from_value::<TenantCreateRequest>(create_request).unwrap_err();
        assert!(
            err.to_string().contains("unknown field `unknown_field`"),
            "expect unknown field `unknown_field` error, got: {err}"
        );
    }

    /// Check that a minimal migrate request with no config results in the expected default settings
    #[test]
    fn test_migrate_request_decode_defaults() {
        let json = r#"{
            "node_id": 123
        }"#;

        let request: TenantShardMigrateRequest = serde_json::from_str(json).unwrap();
        assert_eq!(request.node_id, NodeId(123));
        assert_eq!(request.origin_node_id, None);
        assert!(!request.migration_config.override_scheduler);
        assert!(request.migration_config.prewarm);
        assert_eq!(request.migration_config.secondary_warmup_timeout, None);
        assert_eq!(
            request.migration_config.secondary_download_request_timeout,
            None
        );
    }

    /// Check that a partially specified migration config results in the expected default settings
    #[test]
    fn test_migration_config_decode_defaults() {
        // Specify just one field of the config
        let json = r#"{
        }"#;

        let config: MigrationConfig = serde_json::from_str(json).unwrap();

        // Check each field's expected default value
        assert!(!config.override_scheduler);
        assert!(config.prewarm);
        assert_eq!(config.secondary_warmup_timeout, None);
        assert_eq!(config.secondary_download_request_timeout, None);
        assert_eq!(config.secondary_warmup_timeout, None);

        // Consistency check that the Default impl agrees with our serde defaults
        assert_eq!(MigrationConfig::default(), config);
    }
}
