#pragma once

#include <sys/sysinfo.h>
#include <unistd.h>

#include <ATen/core/ivalue.h>
#include <string>
#include <future>
#include <torch/script.h>
#include <ATen/Parallel.h>
#include "bert_qsl.hpp"
#include "kmp_launcher.hpp"

namespace models {
//
// Functionality:
//   1. Model load&adjust
//
class TorchModel {
public:
  TorchModel (const std::string filename) {

    auto load_model = [filename](int socket){ 
      kmp::KMPAffinityMask mask;
      auto nMaxProc = kmp::KMPLauncher::getMaxProc();
      if (socket == 0) {
        mask.addCore(0).bind();
      } else {
        mask.addCore(nMaxProc - 1).bind();
      }
      
      auto model = torch::jit::load(filename);
      // sched_setaffinity(0, sizeof(backup), &backup);
      return model;
    };

    std::future<torch::jit::script::Module> m0 = std::async(std::launch::async, load_model, 0);
    std::future<torch::jit::script::Module> m1 = std::async(std::launch::async, load_model, 1);

    socket_model_[0] = m1.get();
    socket_model_[1] = m0.get();
  }

  TorchModel ();

  // void load(const std::string filename) {
  //   model_ = torch::jit::load(filename);
  // }

  template <typename... Args>
  at::IValue inference(Args&&... args) {
    return socket_model_[0].forward(std::forward<Args>(args)...);
  }

  template <typename... Args>
  at::IValue inference_at(int socket, Args&&... args) {
    return socket_model_[socket].forward(std::forward<Args>(args)...);
  }

private:
  torch::jit::script::Module socket_model_[2];
};

}
