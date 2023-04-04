#ifndef LOADGEN_TEST_INTF_
#define LOADGEN_TEST_INTF_

#include <vector>

#include <system_under_test.h>
#include <query_sample_library.h>
#include <loadgen.h>

using namespace mlperf;

class SiMaSUT : public SystemUnderTest {
public:

  const std::string& Name();

  void IssueQuery(const std::vector<QuerySample>& samples);

  virtual void FlushQueries();

  void set_sut_name(const std::string & _sut_name) {
    sut_name = _sut_name;
  }

 private:
  std::string sut_name;

};

class SiMaQSL : public QuerySampleLibrary {
 public:
  size_t num_images;

  /// \brief A human readable name for the model.
  const std::string& Name();

  /// \brief Total number of samples in library.
  size_t TotalSampleCount() {
    return num_images;
  }

  /// \brief The number of samples that are guaranteed to fit in RAM.
  size_t PerformanceSampleCount() {
    return num_images;
  }

  /// \brief Loads the requested query samples into memory.
  /// \details Paired with calls to UnloadSamplesFromRam.
  /// In the MultiStream scenarios:
  ///   * Samples will appear more than once.
  ///   * SystemUnderTest::IssueQuery will only be called with a set of samples
  ///     that are neighbors in the vector of samples here, which helps
  ///     SUTs that need the queries to be contiguous.
  /// In all other scenarios:
  ///   * A previously loaded sample will not be loaded again.
  virtual void LoadSamplesToRam(
      const std::vector<QuerySampleIndex>& samples) {
  }

  /// \brief Unloads the requested query samples from memory.
  /// \details In the MultiStream scenarios:
  ///   * Samples may be unloaded the same number of times they were loaded;
  ///     however, if the implementation de-dups loaded samples rather than
  ///     loading samples into contiguous memory, it may unload a sample the
  ///     first time they see it unloaded without a refcounting scheme, ignoring
  ///     subsequent unloads. A refcounting scheme would also work, but is not
  ///     a requirement.
  /// In all other scenarios:
  ///   * A previously unloaded sample will not be unloaded again.
  void UnloadSamplesFromRam(
      const std::vector<QuerySampleIndex>& samples) {
  }

  void set_qsl_name(const std::string & _qsl_name) {
    qsl_name = _qsl_name;
  }

 private:
  std::string qsl_name;

};

#endif // LOADGEN_TEST_INTF_
