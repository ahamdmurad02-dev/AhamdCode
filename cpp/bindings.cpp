#include <cstdint>

extern "C" {
#ifdef _WIN32
__declspec(dllexport)
#endif
std::uint32_t ahamdcode_native_api_version() { return 1; }
}
