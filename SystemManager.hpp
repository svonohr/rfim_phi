#pragma once

#include <fstream>
#include <list>
#include <map>
#include <set>

#include <cstring>
#include <stdint.h>
#include <pthread.h>

class SystemManager
{
public:
  SystemManager();
  ~SystemManager();

  bool OpenFile(const std::string& filename);
  bool CreateFile(const std::string& filename, int dimension, int bytesPerSpin);
  void SetParameters(float kt, float h);

  int GetFileVersion() { return m_head.fileVersion; }
  int GetDimension() { return m_head.dimension; }
  int GetBytesPerSpin() { return m_head.bytesPerSpin; }
  int GetNumberOfSystems() { return m_offsets.size(); }
  float GetTemperature() { return m_head.kt; }
  float GetField() { return m_head.h; }
  size_t GetSystemSize();
  std::set<uint64_t> GetTimes();
  void ListOffsets();

  bool SaveSystem(uint64_t time, const char* system, size_t size);
  bool LoadSystem(uint64_t time, char* system, size_t size);
  friend void* WorkerThread(void* params);

private:
  // Internal structs
  struct FileHeader
  {
    uint32_t signature;
    int32_t fileVersion;
    int32_t dimension;
    int32_t bytesPerSpin;
    float kt;   // temperatur
    float h;    // field
  };
  struct SystemInfo
  {
    size_t offset;
    size_t size;

    SystemInfo()
      : offset(0), size(0) {}
    SystemInfo(size_t offset, size_t size)
      : offset(offset), size(size) {}
  };

  FileHeader m_head;
  std::fstream m_file;
  std::map<uint64_t, SystemInfo> m_offsets;
  std::list<pthread_t> m_worker;
  pthread_mutex_t m_mutex;
};
