#include <iostream>
#include <map>
#include <tuple>
#include <mutex>

#include "loadgen_mem_cfg.h"
#include <simaai/simaai_memory.h>

static int id_to_target [] = {
    SIMAAI_MEM_TARGET_GENERIC,
    SIMAAI_MEM_TARGET_MOSAIC,
    SIMAAI_MEM_TARGET_DMS0,
    SIMAAI_MEM_TARGET_DMS1,
    SIMAAI_MEM_TARGET_DMS2,
    SIMAAI_MEM_TARGET_DMS3,
};

namespace simaai {
namespace mlperf_wrapper {

/**
 * @brief MemoryManager class, this is the SiMa' dram/dram bank memory manager class
 * It uses an algorithm to find the free space for request to use the dram and write/preload 
 * image data onto it and submit the request to mla.
 */
class MemoryManager {
 public:
  MemoryManager() {
    init_memory_slots();
  }
  virtual ~MemoryManager() {};

  /**
   * @brief initialize all the memory slots
   */
  int init_memory_slots ();

  /**
   * @brief Allocate all the slots using memlib
   * @param slot_id as defined in the config
   */
  void allocate_slot(int slot_id);

  /**
   * @brief Initialize non-reserved slots which are used for mapping
   * @param slot_id as defined in the config
   * @param max_size size of the request
   */
  void initialize_slot_available_mappings (int slot_id, size_t max_size);

  /**
   * @brief Update slot mapping after returning a request of size
   * @param slot_id as defined in the config
   * @param used_size size of the dram used
   */
  void update_slot_available_mappings (int slot_id, size_t used_size);

  /**
   * @brief Buffer id to address mapping used to retrieve meta data about memory regions
   * @param slot_id as defined in the config
   * @param buffer_id the buffer id as returned by simamemory library
   */
  void add_buffer_id_mapping (int slot_id, uint64_t buffer_id);

  /**
   * @brief check if slot is available to be used
   * @param request_sz the size of the request memory
   */
  int check_availability(size_t request_sz);

  /**
   * @brief Get_memory used by other interfaces to get a memory regions from manager
   * @param request_sz size of the reuqest
   * @param[out] physical address of the region returned
   * @return virtual address pointer to the memory at some offset
   */
  void * get_memory (size_t request_sz, uint64_t * phys_addr);

  /**
   * @brief getter function for used size
   * @param slot id
   * @return used size or -1
   */  
  size_t get_slot_used_sz (int slot_id);

  /**
   * @brief getter function for maximum available size of slot
   * @param slot id
   * @return max size or -1
   */  
  size_t get_slot_max_sz (int slot_id);

  /**
   * @brief buffer id of the slot
   * @param slot id
   * @return buffer id
   */  
  uint64_t get_slot_bufferid (int slot_id);

  /**
   * @brief Get the physical address of the slot
   * @param slot id
   * @return buffer id
   */  
  uint64_t get_slot_physaddr (int slot_id);

  /**
   * @brief Get the virtual address of the slot
   * @param slot id
   * @return returns virtual address
   */  
  void * get_slot_vaddr (int slot_id);
  size_t get_size(size_t s);

  /**
   * @brief flush_cache, flush the cache lines of the CPU important to do after writing to region returned by manager
   */
  void flush_cache();

  /**
   * @brief Cleanup functions deallocate a slot
   * @param slot_it
   */
  void deallocate_slot (int slot_id);

  /**
   * @brief Cleanup all the slots
   * @param slot_it
   */
  void deinit_memory_slots ();

  /**
   * @brief Remove_id from mappings
   * @param slot_it
   */
  void remove_id_from_map (int slot_id);
 private:
  std::map<int, simaai_memory_t *> m_slot_handle_mappings;
  // [slot_id]::{buffer_id, phys_addr, virtual addr, max_buf_sz}
  std::map<int, std::tuple<uint64_t, uint64_t, void*, size_t>> m_slot_mappings;
  std::mutex mapping_mutex;
  std::mutex query_mutex;
  std::mutex available_mutex;
  // [slot-id]::{maxsize, used}
  std::map<int, std::tuple<size_t, size_t>> m_slot_available_mappings;
};
}
}
