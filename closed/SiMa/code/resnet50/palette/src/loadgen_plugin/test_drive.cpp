
#include <iostream>


#include "loadgen_plugin.h"

#include <loadgen.h>
#include <test_settings.h>

int main() {
    SiMaSUT sima_sut;
    SiMaQSL sima_qsl;

    std::cout << "SUT Name: " << sima_sut.Name() << std::endl;
    std::cout << "SQL Name: " << sima_qsl.Name() << std::endl;

    TestSettings test_settings;
    LogSettings log_settings;

    test_settings.min_query_count = 10;
    test_settings.min_duration_ms = 1;


    mlperf::StartTest(&sima_sut, &sima_qsl, test_settings, log_settings);

    return 0;
}