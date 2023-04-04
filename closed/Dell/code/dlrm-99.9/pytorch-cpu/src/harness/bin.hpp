#pragma once

#include <fstream>
#include <glog/logging.h>

namespace bin {
template <size_t sampleSize>
class BinFile {
private:
    std::string m_Path;
    std::ifstream m_FStream;
    size_t m_FileSize;
    size_t m_Batch;
    std::vector<char> m_Cache;

public:
    BinFile(const std::string& path)
        : m_Path(path), m_FStream(m_Path, std::ifstream::binary) {

        // get file size
        m_FStream.seekg(0, std::ios::end);
        m_FileSize = m_FStream.tellg();
        CHECK_EQ(m_FileSize % sampleSize, 0)
            << "file size is not dividable by sample size" << m_Path;
        m_Batch = m_FileSize / sampleSize;
    }

    ~BinFile() {
        m_FStream.close();
    };

    size_t getFileSize() {
        return m_FileSize;
    }

    size_t getNumSample() {
        return m_Batch;
    }

    // load the entire file
    void loadAll(std::vector<char>& dst) {
        m_FStream.seekg(0, std::ios::beg);
        dst.resize(m_FileSize);
        m_FStream.read(dst.data(), m_FileSize);
        CHECK(m_FStream) << "Unable to parse: " << m_Path;
        CHECK(m_FStream.peek() == EOF) << "Did not consume full file: " << m_Path;
    }

    // cache the entire file
    void cacheAll() {
        loadAll(m_Cache);
    }

    // load only selected indices from the cache, assuming that the first dim is batch dim.
    void loadSamples(std::vector<char>& dst,
                     const std::vector<size_t>& indices) {
        if (m_Cache.empty()) {
            cacheAll();
        }

        dst.resize(sampleSize * indices.size());
        for (size_t i = 0; i < indices.size(); i++) {
            std::memcpy(dst.data() + i * sampleSize,
                        m_Cache.data() + indices[i] * sampleSize,
                        sampleSize);
        }
    }
};

using DLRMBinFile = BinFile<160>;
} // namespace bin
