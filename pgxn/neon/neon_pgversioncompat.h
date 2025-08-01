/*
 * Compatibility macros to cover up differences between supported PostgreSQL versions,
 * to help with compiling the same sources for all of them.
 */

#ifndef NEON_PGVERSIONCOMPAT_H
#define NEON_PGVERSIONCOMPAT_H

#include "fmgr.h"
#include "storage/buf_internals.h"

#if PG_MAJORVERSION_NUM < 16
typedef PGAlignedBlock PGIOAlignedBlock;
#endif

#if PG_MAJORVERSION_NUM < 17
#define NRelFileInfoBackendIsTemp(rinfo) (rinfo.backend != InvalidBackendId)
#else
#define NRelFileInfoBackendIsTemp(rinfo) (rinfo.backend != INVALID_PROC_NUMBER)
#endif

#define RelFileInfoEquals(a, b) ( \
	NInfoGetSpcOid(a) == NInfoGetSpcOid(b) && \
	NInfoGetDbOid(a) == NInfoGetDbOid(b) && \
	NInfoGetRelNumber(a) == NInfoGetRelNumber(b) \
)

/* These macros were turned into static inline functions in v16 */
#if PG_MAJORVERSION_NUM < 16
static inline bool
BufferTagsEqual(const BufferTag *tag1, const BufferTag *tag2)
{
	return BUFFERTAGS_EQUAL(*tag1, *tag2);
}

static inline void
InitBufferTag(BufferTag *tag, const RelFileNode *rnode,
			  ForkNumber forkNum, BlockNumber blockNum)
{
	INIT_BUFFERTAG(*tag, *rnode, forkNum, blockNum);
}
#endif

/* RelFileNode -> RelFileLocator rework */
#if PG_MAJORVERSION_NUM < 16
#define USE_RELFILENODE

#define RELFILEINFO_HDR "storage/relfilenode.h"

#define NRelFileInfo RelFileNode
#define NRelFileInfoBackend RelFileNodeBackend
#define NRelFileNumber Oid

#define InfoFromRelation(rel) (rel)->rd_node
#define InfoFromSMgrRel(srel) (srel)->smgr_rnode.node
#define InfoBFromSMgrRel(srel) (srel)->smgr_rnode
#define InfoFromNInfoB(ninfob) ninfob.node

#define RelFileInfoFmt(rinfo) \
	(rinfo).spcNode, \
	(rinfo).dbNode, \
	(rinfo).relNode

#define RelFileInfoBackendFmt(ninfob) \
	(ninfob).backend, \
	(ninfob).node.spcNode, \
	(ninfob).node.dbNode, \
	(ninfob).node.relNode

#define NInfoGetSpcOid(ninfo)		(ninfo).spcNode
#define NInfoGetDbOid(ninfo)		(ninfo).dbNode
#define NInfoGetRelNumber(ninfo)	(ninfo).relNode

#define CopyNRelFileInfoToBufTag(tag, rinfo) \
	do { \
		(tag).rnode = (rinfo); \
	} while (false)

#define BufTagGetNRelFileInfo(tag) (tag).rnode

#define BufTagGetRelNumber(tagp) ((tagp)->rnode.relNode)

#define BufTagInit(tag, rel_number, fork_number, block_number, spc_oid, db_oid) \
	do { \
		RelFileNode rnode = { .spcNode = (spc_oid), .dbNode = (db_oid), .relNode = (rel_number)}; \
		(tag).forkNum = (fork_number);								\
		(tag).blockNum = (block_number);							\
		(tag).rnode = rnode;										\
	} while (false)

#define InvalidRelFileNumber InvalidOid

#define SMgrRelGetRelInfo(reln)	   	((reln)->smgr_rnode.node)

#define DropRelationAllLocalBuffers DropRelFileNodeAllLocalBuffers

#else							/* major version >= 16 */

#define USE_RELFILELOCATOR

#define RELFILEINFO_HDR "storage/relfilelocator.h"

#define NRelFileInfo RelFileLocator
#define NRelFileInfoBackend RelFileLocatorBackend

#define InfoFromRelation(rel) (rel)->rd_locator
#define InfoFromSMgrRel(srel) (srel)->smgr_rlocator.locator
#define InfoBFromSMgrRel(srel) (srel)->smgr_rlocator
#define InfoFromNInfoB(ninfob) (ninfob).locator

#define RelFileInfoFmt(rinfo) \
	(rinfo).spcOid, \
	(rinfo).dbOid, \
	(rinfo).relNumber
#define RelFileInfoBackendFmt(ninfob) \
	(ninfob).backend, \
	(ninfob).locator.spcOid, \
	(ninfob).locator.dbOid, \
	(ninfob).locator.relNumber

#define NInfoGetSpcOid(ninfo)		(ninfo).spcOid
#define NInfoGetDbOid(ninfo)		(ninfo).dbOid
#define NInfoGetRelNumber(ninfo)	(ninfo).relNumber

#define CopyNRelFileInfoToBufTag(tag, rinfo) \
	do { \
		(tag).spcOid = (rinfo).spcOid; \
		(tag).dbOid = (rinfo).dbOid; \
		(tag).relNumber = (rinfo).relNumber; \
	} while (false)

#define BufTagGetNRelFileInfo(tag) \
	((RelFileLocator) { \
		.spcOid = (tag).spcOid, \
		.dbOid = (tag).dbOid, \
		.relNumber = (tag).relNumber, \
	})

#define BufTagInit(tag, rel_number, fork_number, block_number, spc_oid, db_oid) \
	do { \
		(tag).forkNum = (fork_number);					\
		(tag).blockNum = (block_number);				\
		(tag).spcOid = (spc_oid);						\
		(tag).dbOid = (db_oid);							\
		(tag).relNumber = (rel_number);					\
	} while (false)

#define SMgrRelGetRelInfo(reln)	   	((reln)->smgr_rlocator)

#define DropRelationAllLocalBuffers DropRelationAllLocalBuffers
#endif

#define NRelFileInfoInvalidate(rinfo) do { \
		NInfoGetSpcOid(rinfo) = InvalidOid; \
		NInfoGetDbOid(rinfo) = InvalidOid; \
		NInfoGetRelNumber(rinfo) = InvalidRelFileNumber; \
	} while (0)

#if PG_MAJORVERSION_NUM < 17
#define ProcNumber BackendId
#define INVALID_PROC_NUMBER InvalidBackendId
#define AmAutoVacuumWorkerProcess() (IsAutoVacuumWorkerProcess())
#endif

#if PG_MAJORVERSION_NUM < 17
#define	MyProcNumber (MyProc - &ProcGlobal->allProcs[0])
#endif

#if PG_MAJORVERSION_NUM < 15
extern void InitMaterializedSRF(FunctionCallInfo fcinfo, bits32 flags);
extern TimeLineID GetWALInsertionTimeLine(void);
#endif

/* format codes not present in PG17-; but available in PG18+ */
#define INT64_HEX_FORMAT "%" INT64_MODIFIER "x"
#define UINT64_HEX_FORMAT "%" INT64_MODIFIER "x"

#endif							/* NEON_PGVERSIONCOMPAT_H */
