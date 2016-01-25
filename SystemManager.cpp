#include "SystemManager.hpp"

#include "zlib.h"
#include <iostream>
#include <vector>

// Internal structs
struct SystemBlock1
{
  int32_t time;
};

struct SystemBlock2
{
  uint64_t time;
  uint64_t size;
};

const uint32_t g_signature = 1296647762; // value for 'RFIM'
const int g_fileVersion = 2;
const unsigned int g_maxThreads = 4;

struct ThreadParameters
{
  SystemManager* mgr;
  uint64_t time;
  char* system;
  size_t size;

  ThreadParameters(SystemManager* mgr, uint64_t time, char* system, size_t size)
    : mgr(mgr), time(time), system(system), size(size) {}
};

void* WorkerThread(void* params)
{
  ThreadParameters* p = reinterpret_cast<ThreadParameters*>(params);

  // Compress data
  unsigned long compSize = compressBound(p->size);
  std::vector<char> compSystem(compSize);
  int rc = compress(reinterpret_cast<unsigned char*>(compSystem.data()), &compSize,
                    reinterpret_cast<const unsigned char*>(p->system), p->size);
  delete[] p->system;
  if(rc != Z_OK)
  {
    std::cerr << "Failed compressing system" << std::endl;
    delete p;
    return NULL;
  }

  // Append to file
  pthread_mutex_lock(&p->mgr->m_mutex);
    SystemBlock2 block;
    block.time = p->time;
    block.size = compSize;
    p->mgr->m_file.seekp(0, std::ios::end);
    p->mgr->m_file.write(reinterpret_cast<char*>(&block), sizeof(block));
    p->mgr->m_file.write(compSystem.data(), compSize);
  pthread_mutex_unlock(&p->mgr->m_mutex);
  delete p;
  return NULL;
}

SystemManager::SystemManager()
{
  m_head.signature = g_signature;
  m_head.fileVersion = g_fileVersion;
  m_head.dimension = 0;
  m_head.bytesPerSpin = 0;
  m_head.kt = 0;
  m_head.h = 0;

  pthread_mutex_init(&m_mutex, NULL);
}

SystemManager::~SystemManager()
{
  // Wait for threads to finish
  std::list<pthread_t>::iterator it;
  for(it = m_worker.begin(); it != m_worker.end(); ++it)
    pthread_join(*it, NULL);
  m_worker.clear();
  pthread_mutex_destroy(&m_mutex);
}

bool SystemManager::OpenFile(const std::string& filename)
{
  // Open file
  m_file.open(filename.c_str(), std::ios::in | std::ios::binary);
  if(!m_file.is_open())
  {
    std::cerr << "Unable to open file: " << filename << std::endl;
    return false;
  }

  // Read header
  m_file.read(reinterpret_cast<char*>(&m_head), sizeof(m_head));
  if(!m_file.good())
  {
    std::cerr << "Unable to read header: " << filename << std::endl;
    return false;
  }

  // Check header
  if(m_head.signature != g_signature ||
     m_head.fileVersion < 0 ||
     m_head.dimension < 0 ||
     m_head.bytesPerSpin < 0)
  {
    std::cerr << "Invalid header: " << filename << std::endl;
    return false;
  }
  if(m_head.fileVersion > g_fileVersion)
  {
    std::cerr << "File was create with a newer version: " << filename << std::endl;
    return false;
  }

  // Read blocks
  if(m_head.fileVersion == 2)
  {
    // Read version 2 blocks
    SystemBlock2 block;
    m_file.read(reinterpret_cast<char*>(&block), sizeof(block));
    while(m_file.good())
    {
      m_offsets[block.time] = SystemInfo(m_file.tellg(), block.size);
      m_file.seekg(block.size, std::ios::cur);
      m_file.read(reinterpret_cast<char*>(&block), sizeof(block));
    }
  }
  else
  {
    // Read version 1 blocks
    SystemBlock1 block;
    size_t blockSize = this->GetSystemSize();
    m_file.read(reinterpret_cast<char*>(&block), sizeof(block));
    while(m_file.good())
    {
      m_offsets[block.time] = SystemInfo(m_file.tellg(), blockSize);
      m_file.seekg(blockSize, std::ios::cur);
      m_file.read(reinterpret_cast<char*>(&block), sizeof(block));
    }
  }
  m_file.clear();
  return true;
}

bool SystemManager::CreateFile(const std::string& filename, int dimension, int bytesPerSpin)
{
  // Create new file
  m_file.open(filename.c_str(), std::ios::out | std::ios::binary | std::ios::trunc);
  if(!m_file.is_open())
  {
    std::cerr << "Unable to create file: " << filename << std::endl;
    return false;
  }

  // Write header
  m_head.dimension = dimension;
  m_head.bytesPerSpin = bytesPerSpin;
  m_file.write(reinterpret_cast<char*>(&m_head), sizeof(m_head));
  return true;
}

void SystemManager::SetParameters(float kt, float h)
{
  // Rewrite header
  m_head.kt = kt;
  m_head.h = h;

  pthread_mutex_lock(&m_mutex);
    m_file.seekp(0, std::ios::beg);
    m_file.write(reinterpret_cast<char*>(&m_head), sizeof(m_head));
  pthread_mutex_unlock(&m_mutex);
}

size_t SystemManager::GetSystemSize()
{
  size_t dim = this->GetDimension();
  return dim*dim*dim*this->GetBytesPerSpin();
}

std::set<uint64_t> SystemManager::GetTimes()
{
  std::set<uint64_t> out;
  std::map<uint64_t, SystemInfo>::iterator it;
  for(it = m_offsets.begin(); it != m_offsets.end(); ++it)
    out.insert(it->first);
  return out;
}

void SystemManager::ListOffsets()
{
  std::map<uint64_t, SystemInfo>::iterator it;
  for(it = m_offsets.begin(); it != m_offsets.end(); ++it)
    std::cout << "System for time " << it->first << " at offset " << it->second.offset << std::endl;
}

bool SystemManager::SaveSystem(uint64_t time, const char* system, size_t size)
{
  // Check size
  if(size != this->GetSystemSize())
  {
    std::cerr << "System size is wrong" << std::endl;
    return false;
  }

  // Find block
  std::map<uint64_t, SystemInfo>::iterator it = m_offsets.find(time);
  if(it != m_offsets.end())
  {
    std::cerr << "Time already stored in file: " << time << std::endl;
    return false;
  }

  // Wait if to many threads are running
  if(m_worker.size() >= g_maxThreads)
  {
    pthread_join(m_worker.front(), NULL);
    m_worker.pop_front();
  }

  // Copy data and start thread
  char* copy = new char[size];
  memcpy(copy, system, size);

  pthread_t thread;
  ThreadParameters* param = new ThreadParameters(this, time, copy, size);
  pthread_create(&thread, NULL, WorkerThread, param);
  m_worker.push_back(thread);
  return true;
}

bool SystemManager::LoadSystem(uint64_t time, char* system, size_t size)
{
  // Check size
  if(size != this->GetSystemSize())
  {
    std::cerr << "System size is wrong" << std::endl;
    return false;
  }

  // Find block
  std::map<uint64_t, SystemInfo>::iterator it = m_offsets.find(time);
  if(it == m_offsets.end())
  {
    std::cerr << "Time not stored in file: " << time << std::endl;
    return false;
  }

  // Read block
  if(m_head.fileVersion == 2)
  {
    // Read version 2 system
    std::vector<char> compSystem(it->second.size);
    m_file.seekg(it->second.offset, std::ios::beg);
    m_file.read(compSystem.data(), compSystem.size());

    // Uncompress data
    int rc = uncompress(reinterpret_cast<unsigned char*>(system), &size,
                        reinterpret_cast<const unsigned char*>(compSystem.data()), compSystem.size());
    if(rc != Z_OK)
    {
      std::cerr << "Failed uncompressing system" << std::endl;
      return false;
    }
  }
  else
  {
    // Read version 1 system
    m_file.seekg(it->second.offset, std::ios::beg);
    m_file.read(system, it->second.size);
  }
  return true;
}
