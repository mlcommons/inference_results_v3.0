#ifndef IMAGE_LOADER_H_
#define IMAGE_LOADER_H_

#include <cstring>
#include <cstdint>
#include <memory>
#include <iostream>
#include <random>
#include <vector>

#include "loadgen_mem_manager.h"
// #include <map>

namespace simaai {
namespace mlperf_wrapper {
/**
 * @brief ImageLoader class to preload sample images into dram
 */
template<typename T>
class ImageLoader {
 public:
  ImageLoader() {};
  ImageLoader (const std::string & sample_data_fpath, int _num_images)
      :stride(3),
       height(224),
       width(224),
       num_images(_num_images),
       img_idx(0) {
    image_fpath = sample_data_fpath;
    images.reserve(_num_images);
    m_manager = new simaai::mlperf_wrapper::MemoryManager();
  };

  ~ImageLoader() {
    unload();
  };

  /**
   * @brief Helper API to return number of samples loaded to DRAM
   */
  int64_t get_num_of_images() { return num_images; }
  int64_t get_num_of_test_images() { return images.size(); }

  /**
   * @brief Helper API to return number at a position
   */
  void *  get_image_at_pos (int64_t pos) { return images[pos]; }

  /**
   * @brief Helper API to get image size
   */
  size_t get_cur_image_size() { return cur_sample_size; }

  /**
   * @brief Load API, to load images to dram
   * @return TRUE on success, FALSE on failure
   */
  bool load () {
    FILE * fp = fopen(image_fpath.c_str(), "r+");
    if (!fp) {
      std::cout << "--- MLPerf:ImageLoader:ERROR Unable to open file "<< image_fpath.c_str();
      return false;
    }

    size_t sz = stride * height * width;
    if (sz == 0) {
      std::cout << "--- MLPerf:ImageLoader::ERROR Size cannot be zero";
      return false;
    }

    int cnt = 0;
    size_t bytes_read = 0;

    do  {
      img_data = new T[stride * height * width];

      bytes_read = fread(img_data, sizeof(int8_t), sz, fp);
      if (bytes_read <= 0) {
        break;
      }
      cur_sample_size = bytes_read;
      images.push_back(img_data);
      // images.insert({img_idx, img_data});
      // img_idx++;
    } while (bytes_read != 0);

    fclose(fp);
    print_info();
    return true;
  };

  /**
   * @brief Load API, to load images to dram
   * @return TRUE on success, FALSE on failure
   */
  bool load_batch (size_t batch_size) {
    FILE * fp = fopen(image_fpath.c_str(), "r+");
    if (!fp) {
      std::cout << "--- MLPerf:ImageLoader:ERROR Unable to open file "<< image_fpath.c_str();
      return false;
    }

    size_t sz = stride * height * width;
    if (sz == 0) {
      std::cout << "--- MLPerf:ImageLoader::ERROR Size cannot be zero";
      return false;
    }

    int cnt = 0;
    size_t bytes_read = 0;
    uint32_t img_idx = 0;
    
    std::cout << "--- MLPerf::ImageLoader:: Loading the images to dram" << num_images << "\n";
    
    do  {
      if ((img_idx + 1) > num_images)
        break;

      uint64_t phys_addr;
      int8_t * data_ptr = static_cast<int8_t *>(m_manager->get_memory(sz, &phys_addr));
      bytes_read = fread(data_ptr, sizeof(int8_t), sz, fp);
      if (bytes_read <= 0) {
        break;
      }
      cur_sample_size = bytes_read;
      images.push_back(img_data);
      m_image_addr_map[img_idx++] = phys_addr;
    } while (bytes_read != 0);

    std::cout << "---- MLperf:ImageLoader:: Preloading to dram done\n";
    m_manager->flush_cache();
    fclose(fp);
    print_info();
    return true;
  };

  uint64_t get_image_phys_addr(int32_t image_id) {
    if (m_image_addr_map.size() <= 0)
      throw std::runtime_error("--MLPERF::[FATAL] Image data is not loaded into DRAM");
    return m_image_addr_map[image_id];
  }
  
  /**
   * @brief Helper API to unload images/cleanup
   */
  void unload () {
    images.clear();
  }

  void print_info() {
    std::cout << "--- MLPerf::ImageLoader:: Loaded Sample data from " << image_fpath.c_str() << "\n";
    std::cout << "--- MLPerf::ImageLoader:: Loaded " << get_num_of_test_images() << " samples into dram\n";
    std::cout << "--- MLPerf::ImageLoader:: Image is of type " << typeid(T).name() << "\n";
    std::cout << "--- MLPerf::ImageLoader:: Total size of sample data loaded " << get_num_of_test_images() * cur_sample_size << "\n";
    std::cout << "--- MLPerf::ImageLoader:: Ready to run\n";

    std::cout << "--- MLPerf::Batch load details, number of images " << m_image_addr_map.size() << "\n";
    // for (int i = 0; i < m_image_addr_map.size(); i++)
    //   fprintf(stderr, "---MLPerf:: img_idx[%d]:physaddr[0x%x]\n", i, m_image_addr_map[i]);
  }

  void debug_test () {
    std::cout << "Number of testing samples " << get_num_of_test_images() << "\n";
    if (get_num_of_test_images() > 0) {
      for (int n = 0; n < 10; ++n) {
        auto i = static_cast<int8_t *>(get_image_at_pos(n));

        std::string name("foo.bin");
        auto fname = name + std::to_string(n);

        std::cout << "File name " << fname << "\n";

        FILE * _fp = fopen(fname.c_str(), "w+");
        fwrite(i, sizeof(int8_t), 150528, _fp);
        fclose(_fp);
      }
    }
  }

 private:
  size_t num_images, cur_sample_size;
  int width, height, stride;
  std::string image_fpath;
  std::vector<T *> images;
  // std::map<int, T*> images;
  int img_idx;
  T *img_data;
  simaai::mlperf_wrapper::MemoryManager * m_manager;
  std::map<int, uint64_t> m_image_addr_map;
};
}
}

#endif // IMAGE_LOADER_H_
