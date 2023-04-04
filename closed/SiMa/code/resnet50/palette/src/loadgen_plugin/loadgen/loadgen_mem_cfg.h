#ifndef LOADGEN_MEM_CFG_H_
#define LOADGEN_MEM_CFG_H_

#include <simaai/simaai_memory.h>

#define MAX_SLOT_NAME 256
#define MAX_MEM_REGIONS 24
#define TOY_MAX_MEM_REGIONS 6
#define MAX_RESERVED_MEM_REGIONS 5

typedef struct loadgen_mem_cfg_s {
    int idx;  /// < An index or unique id for the slot, readable by loadgen
    char slot_name[MAX_SLOT_NAME]; /// < Just a 'name'
    int target_id;   /// < simaai memory target id
    size_t max_size; /// < Maximum size as bytes
    bool is_reserved; /// < Is the memory usable i.e non-reserved.
} loadgen_mem_cfg_t ;

/* 
 * SiMa's SoC has a 16GB ddr, the utilization the DDR is managed by
 * sima-memory driver and sima-memory userspace library.
 * For mlperf we use 7.5GB of DRAM to preload all the sample images to
 * DDR and work with it.
 * Below is the definition of all the slots available from the 4DMA
 * banks addressable to the SiMa MLA
 */

static loadgen_mem_cfg_t mem_cfg_full_mode[MAX_MEM_REGIONS] = {
/* From DMS0 */
    {0, "reservedslot0", SIMAAI_MEM_TARGET_DMS0, 256, true},
    {1, "slot0",         SIMAAI_MEM_TARGET_DMS0, 500, false},
    {2, "slot1",         SIMAAI_MEM_TARGET_DMS0, 500, false},
    {3, "slot2",         SIMAAI_MEM_TARGET_DMS0, 250, false},
/* From DMS1 */
    {4, "slot3",         SIMAAI_MEM_TARGET_DMS1, 500, false},
    {5, "reservedslot1", SIMAAI_MEM_TARGET_DMS1, 256, true},
    {6, "slot4",         SIMAAI_MEM_TARGET_DMS1, 500, false},
    {7, "slot5",         SIMAAI_MEM_TARGET_DMS1, 500, false},
    {9, "slot6",         SIMAAI_MEM_TARGET_DMS1, 250, false},
/* From DMS2 */
    {10, "slot7",        SIMAAI_MEM_TARGET_DMS2, 512, false},
    {11, "slot8",        SIMAAI_MEM_TARGET_DMS2, 512, false},
    {12, "reservedslot2",SIMAAI_MEM_TARGET_DMS2, 256, true},
    {13, "slot9",        SIMAAI_MEM_TARGET_DMS2, 512, false},
    {14, "slot10",       SIMAAI_MEM_TARGET_DMS2, 512, false},
    {15, "slot11",       SIMAAI_MEM_TARGET_DMS2, 256, false},
/* /\* From DMS3 *\/ */
    {16, "reservedslot3",SIMAAI_MEM_TARGET_DMS3, 256, true},
    {17, "slot12",       SIMAAI_MEM_TARGET_DMS3, 512, false},
    {18, "slot13",       SIMAAI_MEM_TARGET_DMS3, 512, false},
    {19, "slot14",       SIMAAI_MEM_TARGET_DMS3, 512, false},
    {20, "slot15",       SIMAAI_MEM_TARGET_DMS3, 512, false},
    {21, "slot16",       SIMAAI_MEM_TARGET_DMS3, 256, false},
    {-1, "", -1, 0, false},
};

#endif // LOADGEN_MEM_CFG_H_
