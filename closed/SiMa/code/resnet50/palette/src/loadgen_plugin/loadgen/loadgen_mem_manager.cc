#include <iostream>

#include "loadgen_mem_cfg.h"
#include "loadgen_mem_manager.h"
#include <simaai/simaai_memory.h>

namespace simaai {
namespace mlperf_wrapper {

int MemoryManager::init_memory_slots () {
  for (int i = 0; i < MAX_MEM_REGIONS; i++) {
    if (mem_cfg_full_mode[i].idx == -1)
      break;

    allocate_slot(i);
    std::cout << "--- MLPerf::[memory] Allocated memory for slot_id " << mem_cfg_full_mode[i].idx << "\n";
  }
  std::cout << "--- MLPerf:: Number of available slots " << m_slot_mappings.size() << "\n";
  return 0;
};

size_t MemoryManager::get_size(size_t s) {
  return s * 1024 * 1024;
}
    
void MemoryManager::allocate_slot (int idx) {
  if (idx < 0)
    throw std::runtime_error("Undefined slot id");

  size_t mem_sz = get_size(mem_cfg_full_mode[idx].max_size);
  simaai_memory_t * m_handle = simaai_memory_alloc_flags(mem_sz,
                                                         mem_cfg_full_mode[idx].target_id,
                                                         SIMAAI_MEM_FLAG_CACHED);


  if (!m_handle) {
    std::cout << "Unable to allocate requested memory slot_id " << idx  << " size "
              << mem_cfg_full_mode[idx].max_size << "\n";
    throw std::runtime_error("alloc failed");
  }

  // Get more details, like buffer id and physicall address here
  if (!mem_cfg_full_mode[idx].is_reserved) {
    int slot_id = mem_cfg_full_mode[idx].idx;
    add_buffer_id_mapping (slot_id, simaai_memory_get_id(m_handle));
  }
};

void MemoryManager::add_buffer_id_mapping (int slot_id, uint64_t buffer_id) {
  std::lock_guard<std::mutex> _lock(mapping_mutex);
  void *res;
  uint64_t phys_addr;
  size_t size;
  
  auto search = m_slot_mappings.find(slot_id);

  if (search == m_slot_mappings.end()) {
    simaai_memory_t * m = simaai_memory_attach(buffer_id);
    res = simaai_memory_map(m);
    phys_addr = simaai_memory_get_phys(m);
    size = simaai_memory_get_size(m);

    // Slot details
    m_slot_mappings[slot_id] = std::make_tuple(buffer_id, phys_addr, res, size);
    m_slot_handle_mappings[slot_id] = m;
    initialize_slot_available_mappings (slot_id, size);
  
    // fprintf(stderr, "Allocated memory of size [%ld], with buffer_id: [%ld],"
    //         "starting a dram addres: [0x%x]:[%p]\n", size, buffer_id, phys_addr);
  } else {
    std::cout << "--- MLPerf::[Memory] slot_id initialized\n";
  }

  return;
};

void MemoryManager::remove_id_from_map (int slot_id) {
  std::lock_guard<std::mutex> _lock(mapping_mutex);

  auto search = m_slot_mappings.find(slot_id);

  if (search != m_slot_mappings.end()) {
    std::cout << "Erased memory mapping\n";
    m_slot_mappings.erase(search);
  }
  else
    std::cout<< "slot id Not found";
  return;
};

int MemoryManager::check_availability(size_t request_sz) {
  std::lock_guard<std::mutex> _lock(query_mutex);
  for(auto const& m: m_slot_available_mappings) {
    size_t max = get_slot_max_sz(m.first);
    size_t used = get_slot_used_sz(m.first);
    size_t available = (max - used);

    if (available >= request_sz) {
      return m.first;
    }
  }

  return -1;
};

// Search for a free slot, return slot-id & virtual address
void * MemoryManager::get_memory (size_t request_sz, uint64_t * phys_addr) {
  // Check if this size can be served by us.
  // If 'yes' then return, vaddr, & slot_id
  int slot_id = check_availability(request_sz);
  
  if (slot_id < 0)
    throw std::runtime_error("Memory exhausted killing self");

  // std::cout << "--- MLPerf::[Memory] Request allocated from slot_id " << slot_id << "\n";
  // Got a slot id here. From the slot id get some metadata to return
  std::lock_guard<std::mutex> _lock(query_mutex);

  uint64_t buf_id  = get_slot_bufferid(slot_id);
  uint64_t _phys_addr  = get_slot_physaddr(slot_id);
  size_t maxsize  = get_slot_max_sz(slot_id);

  *phys_addr = _phys_addr  + get_slot_used_sz(slot_id);
  void * vaddr = get_slot_vaddr(slot_id) + get_slot_used_sz(slot_id);
  update_slot_available_mappings (slot_id, request_sz);

  // fprintf(stderr, "---- MLPerf::[DEBUG] request_sz: %ld, satisfied at slot_id : %d, "
  //         "starting at physaddr:[0x%x], writable at physaddr[0x%x]\n", request_sz, slot_id, _phys_addr, *phys_addr);

  return vaddr;
};

void MemoryManager::initialize_slot_available_mappings (int slot_id, size_t max_size) {
  std::lock_guard<std::mutex> _lock(available_mutex);
  m_slot_available_mappings[slot_id] = std::make_tuple(max_size, 0);
};

void MemoryManager::update_slot_available_mappings (int slot_id, size_t used_size) {
  std::lock_guard<std::mutex> _lock(available_mutex);
  std::get<1>(m_slot_available_mappings[slot_id]) += used_size;
};

size_t MemoryManager::get_slot_used_sz (int slot_id) {
  std::lock_guard<std::mutex> _lock(available_mutex);
  auto search = m_slot_available_mappings.find(slot_id);

  if (search == m_slot_available_mappings.end()) {
    std::cout << "--- MLPerf::[ERROR], slot unitialized " << slot_id << "\n";
    throw std::runtime_error("slot uninitizlied");
  } else {
    return std::get<1>(m_slot_available_mappings[slot_id]);
  }

  return -1;
};

uint64_t MemoryManager::get_slot_bufferid (int slot_id) {
  std::lock_guard<std::mutex> _lock(mapping_mutex);
  auto search = m_slot_mappings.find(slot_id);

  if (search == m_slot_mappings.end()) {
    std::cout << "--- MLPerf::[ERROR], slot unitialized\n";
    throw std::runtime_error("slot uninitizlied");
  } else {
    return std::get<0>(m_slot_mappings[slot_id]);
  }

  return -1;
};

uint64_t MemoryManager::get_slot_physaddr (int slot_id) {
  std::lock_guard<std::mutex> _lock(mapping_mutex);
  auto search = m_slot_mappings.find(slot_id);

  if (search == m_slot_mappings.end()) {
    std::cout << "--- MLPerf::[ERROR], slot unitialized\n";
    throw std::runtime_error("slot uninitizlied");
  } else {
    return std::get<1>(m_slot_mappings[slot_id]);
  }

  return NULL;
};

void * MemoryManager::get_slot_vaddr (int slot_id) {
  std::lock_guard<std::mutex> _lock(mapping_mutex);
  auto search = m_slot_mappings.find(slot_id);

  if (search == m_slot_mappings.end()) {
    std::cout << "--- MLPerf::[ERROR], slot unitialized\n";
    throw std::runtime_error("slot uninitizlied");
  } else {
    return std::get<2>(m_slot_mappings[slot_id]);
  }

  return NULL;
};

size_t MemoryManager::get_slot_max_sz (int slot_id) {
  std::lock_guard<std::mutex> _lock(mapping_mutex);
  auto search = m_slot_mappings.find(slot_id);

  if (search == m_slot_mappings.end()) {
    std::cout << "--- MLPerf::[ERROR], slot unitialized\n";
    throw std::runtime_error("slot uninitizlied");
  } else {
    return std::get<3>(m_slot_mappings[slot_id]);
  }

  return -1;
};

void MemoryManager::deallocate_slot (int slot_id) {
  if (slot_id < 0)
    throw std::runtime_error("Undefined slot id");

  simaai_memory_t * m = m_slot_handle_mappings[slot_id];
  simaai_memory_unmap(m);
  simaai_memory_free(m);
  // Get more details, like buffer id and physicall address here
};

void MemoryManager::deinit_memory_slots () {
  for (auto const& m: m_slot_available_mappings)
    deallocate_slot(m.first);
};

void MemoryManager::flush_cache() {
  for(auto const& m: m_slot_handle_mappings) 
    simaai_memory_flush_cache(m.second);
};
}
}
