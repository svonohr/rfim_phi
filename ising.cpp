#pragma offload_attribute(push, target(mic))

#include "SystemManager.hpp"

#include <iostream>
#include <set>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <immintrin.h>
#include <omp.h>

#ifdef __MIC__
#include <micvec.h>
#endif

#define ALLOC alloc_if(1) free_if(0)
#define REUSE alloc_if(0) free_if(0)
#define FREE  alloc_if(0) free_if(1)

uint32_t expBeta[14];
const unsigned int N = 256;   // System side length
const unsigned int BX = 256;  // Block sizes
const unsigned int BY = 4;
const unsigned int BZ = 4;

// Philox RNG for Xeon Phi cards
__forceinline
void philox2x32_mic(uint64_t counter, uint32_t key, __m512i& rnd1, __m512i& rnd2)
{
#ifdef __MIC__
  const __m512i m = _mm512_set1_epi32(0xD256D193);
  const __m512i w = _mm512_set1_epi32(0x9E3779B9);
  const __m512i incr = _mm512_set_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);

  __m512i r = _mm512_set1_epi32(counter & 0xFFFFFFFF);
  __m512i l = _mm512_set1_epi32(counter >> 32);
  __m512i keyV = _mm512_set1_epi32(key);
  keyV = _mm512_add_epi32(keyV, incr);

  #pragma unroll(10)
  for(int i = 0; i < 10; ++i)
  {
    __m512i l_old = l;
    l = _mm512_mullo_epi32(r, m);
    r = _mm512_xor_epi32(_mm512_xor_epi32(_mm512_mulhi_epu32(r, m), keyV), l_old);
    keyV = _mm512_add_epi32(keyV, w);
  }
  rnd1 = r;
  rnd2 = l;
#endif
}

// Main spin update routine
__forceinline
void spinFlipCore(uint32_t* updated, const uint32_t* neighbours, const uint32_t* field,
                  unsigned int x, unsigned int y, unsigned int z, __m512i rndInt)
{
#ifdef __MIC__
  const __m512i one = _mm512_set1_epi32(0xFFFFFFFF);
  const __m512i zero = _mm512_setzero_epi32();

  // calculate indices
  unsigned int x0 = (x+N-1)%N;
  unsigned int y0 = (y+N-1)%N;
  unsigned int z0 = (z+N-1)%N;
  unsigned int x1 = (x+17)%N;
  unsigned int y1 = (y+1)%N;
  unsigned int z1 = (z+1)%N;

  // neighbour spins
  Iu32vec16 n[6];
  n[0] = _mm512_loadunpacklo_epi32(_mm512_undefined_epi32(), neighbours + z*N*N + y*N + x0);
  n[0] = _mm512_loadunpackhi_epi32(n[0], neighbours + z*N*N + y*N + x + 16 - 1);
  n[1] = _mm512_loadunpacklo_epi32(_mm512_undefined_epi32(), neighbours + z*N*N + y*N + x + 1);
  n[1] = _mm512_loadunpackhi_epi32(n[1], neighbours + z*N*N + y*N + x1);
  n[2] = _mm512_load_epi32(neighbours + z*N*N + y0*N + x);
  n[3] = _mm512_load_epi32(neighbours + z*N*N + y1*N + x);
  n[4] = _mm512_load_epi32(neighbours + z0*N*N + y*N + x);
  n[5] = _mm512_load_epi32(neighbours + z1*N*N + y*N + x);

  // bits are set if spins are antiparallel
  unsigned int i = z*N*N + y*N + x;
  Iu32vec16 current = _mm512_load_epi32(updated + i);
  #pragma unroll(6)
  for(int j = 0; j < 6; ++j)
    n[j] = current ^ n[j];

  // count wrong spins using vertical counters
  Iu32vec16 c0, c1, c2, carry;

  c0 = n[0] ^ n[1];
  c1 = n[0] & n[1];

  c0 ^= n[2];
  c1 |= andn(c0, n[2]);

  c0 ^= n[3];
  carry = andn(c0, n[3]);
  c1 ^= carry;
  c2 = andn(c1, carry);

  c0 ^= n[4];
  carry = andn(c0, n[4]);
  c1 ^= carry;
  c2 |= andn(c1, carry);

  c0 ^= n[5];
  carry = andn(c0, n[5]);
  c1 ^= carry;
  c2 |= andn(c1, carry);

  Iu32vec16 w1 = andn(c2, andn(c1, c0));
  Iu32vec16 w2 = andn(c2, andn(c0, c1));
  Iu32vec16 w3 = andn(c2, c0 & c1);
  Iu32vec16 w4 = andn(c0, andn(c1, c2));
  Iu32vec16 w5 = andn(c1, c0 & c2);
  Iu32vec16 w6 = andn(c0, c1 & c2);

  // relation to field
  Iu32vec16 e[7];
  Iu32vec16 f = current ^ _mm512_load_epi32(field + i);
  #pragma unroll(7)
  for(int j = 0; j < 7; j++)
  {
    __mmask16 ep = _mm512_cmple_epu32_mask(rndInt, _mm512_set1_epi32(expBeta[2*j]));
    __mmask16 em = _mm512_cmple_epu32_mask(rndInt, _mm512_set1_epi32(expBeta[2*j+1]));
    e[6-j] = _mm512_mask_mov_epi32(_mm512_mask_mov_epi32(zero, em, f), ep, one);
  }

  // check for spin flip
  Iu32vec16 flip = e[0] | e[1] & w1 | e[2] & w2 | e[3] & w3 | e[4] & w4 | e[5] & w5 | e[6] & w6;
  _mm512_store_epi32(updated + i, flip ^ current);
#endif
}

void spinFlip(uint32_t* updated, const uint32_t* neighbours, const uint32_t* field, uint64_t counter)
{
  #pragma omp parallel for collapse(3) schedule(static)
  for(unsigned int bz = 0; bz < N; bz+=BZ)
  for(unsigned int by = 0; by < N; by+=BY)
  for(unsigned int bx = 0; bx < N; bx+=BX)
  {
    for(unsigned int z = 0; z < BZ; ++z)
    for(unsigned int y = 0; y < BY; ++y)
    for(unsigned int x = 0; x < BX; x+=32)
    {
      __m512i rndInt1, rndInt2;
      philox2x32_mic(counter, (bz+z)*N*N + (by+y)*N + bx+x, rndInt1, rndInt2);

      // update block of 32 spins
      spinFlipCore(updated, neighbours, field, bx+x, by+y, bz+z, rndInt1);
      spinFlipCore(updated, neighbours, field, bx+x+16, by+y, bz+z, rndInt2);
    }
  }
}

void randomize(uint32_t* system, uint64_t seed)
{
  #pragma omp parallel for collapse(3) schedule(static)
  for(unsigned int bz = 0; bz < N; bz+=BZ)
  for(unsigned int by = 0; by < N; by+=BY)
  for(unsigned int bx = 0; bx < N; bx+=BX)
  {
    for(unsigned int z = 0; z < BZ; ++z)
    for(unsigned int y = 0; y < BY; ++y)
    for(unsigned int x = 0; x < BX; x+=32)
    {
      __m512i rnd1, rnd2;
      unsigned int i = (bz+z)*N*N + (by+y)*N + bx+x;
      philox2x32_mic(seed, i, rnd1, rnd2);
      _mm512_store_epi32(system + i, rnd1);
      _mm512_store_epi32(system + i + 16, rnd2);
    }
  }
}

void setParameters(float temperature, float h)
{
  float beta = 1.f/temperature;
  for(int i = 0; i < 14; i+=2)
  {
    float expBetaP = exp(-2*beta*(i-6+h));
    float expBetaM = exp(-2*beta*(i-6-h));

    // Convert to int
    if(expBetaP >= 1)
      expBeta[i] = UINT32_MAX;
    else
      expBeta[i] = expBetaP*UINT32_MAX;

    if(expBetaM >= 1)
      expBeta[i+1] = UINT32_MAX;
    else
      expBeta[i+1] = expBetaM*UINT32_MAX;
  }
}

double get_cputime()
{
  timespec tp;
  clock_gettime(CLOCK_MONOTONIC, &tp);
  return tp.tv_sec + tp.tv_nsec*1E-9;
}

// Alignment and offset of memory storing the system and random field
// See https://software.intel.com/en-us/forums/intel-many-integrated-core/topic/540363 for discussion
const unsigned int ALIGN = 16*1024*1024;
const unsigned int OFFSET = 32*1024;

#pragma offload_attribute(pop)

int main(int argc, char** argv)
{
  // Parameters
  std::string filename = "system.dat";
  float temperature = 1.0f;
  float epsilon = 0;
  uint64_t seed = 0;
  int phi_card = 0;

  // Get parameters
  for(int i = 1; i < argc; i++)
  {
    if(strlen(argv[i]) != 2 || argv[i][0] != '-' || i+1 == argc)
    {
      std::cout << "Invalid option " << argv[i] << std::endl;
      return EXIT_FAILURE;
    }

    switch(argv[i][1])
    {
      case 'o': // Output file
        filename = argv[i+1];
        break;
      case 't': // Temperature
        temperature = std::stof(argv[i+1]);
        break;
      case 'e': // Epsilon
        epsilon = std::stof(argv[i+1]);
        break;
      case 's': // RNG Seed
        seed = std::stoull(argv[i+1]);
        break;
      case 'p': // Numer of Phi card to use
        phi_card = std::stoi(argv[i+1]);
        break;
      default:
        std::cout << "Invalid option " << argv[i] << std::endl;
        return EXIT_FAILURE;
    }
    i++;
  }

  // Required timesteps
  std::set<uint64_t> times;
  uint64_t maxT = 10000000;
  for(uint64_t j = 1; j <= maxT/10; j*=10)
  {
    times.insert(j);
    for(uint64_t i = 1; i < maxT; i+=(i+1)/2)
      times.insert(j+i);
  }
  size_t system_size = 2*4*N*N*N;
  std::cout << "Timesteps to save: " << times.size() << std::endl;
  std::cout << "Space required: " << times.size()*system_size/1024/1024 << " MB" << std::endl;

  // Info
  float field = epsilon * temperature;
  std::cout << "Temperature: " << temperature << std::endl;
  std::cout << "Field: " << field << std::endl;
  std::cout << "Seed: " << seed << std::endl;
  std::cout << "Dimension: " << N << "^3" << std::endl;

  // Initialize
  uint32_t* systemState = (uint32_t*)_mm_malloc(system_size, 64);
  uint32_t* system0;
  #pragma offload target(mic:phi_card)       \
    in(systemState : length(2*N*N*N) ALLOC)  \
    nocopy(system0)
  {
    system0 = (uint32_t*)_mm_malloc(sizeof(uint32_t)*(N*N*N*4 + OFFSET), ALIGN);
    uint32_t* system1 = system0 + N*N*N + OFFSET/4;
    uint32_t* field0 = system1 + N*N*N + OFFSET/4;
    uint32_t* field1 = field0 + N*N*N + OFFSET/4;
    setParameters(temperature, field);

    randomize(system0, seed++);
    randomize(system1, seed++);
    randomize(field0, seed++);
    randomize(field1, seed++);
  }

  // Open output file
  SystemManager mgr;
  mgr.CreateFile(filename, N, 2*4);
  mgr.SetParameters(temperature, field);

  // Set precision
  std::cout.precision(2);
  std::cout << std::fixed;

  // Start simulation
  times.insert(0);
  auto prev = times.begin();
  uint64_t maxTime = *times.rbegin();
  double startTime = get_cputime();
  for(auto it = std::next(prev); it != times.end(); ++it)
  {
    // Necessary steps for next system
    uint64_t steps = *it - *prev;
    prev = it;

    // Do steps
    #pragma offload target(mic:phi_card)        \
      out(systemState : length(2*N*N*N) REUSE)  \
      nocopy(system0)
    {
      uint32_t* system1 = system0 + N*N*N + OFFSET/4;
      uint32_t* field0 = system1 + N*N*N + OFFSET/4;
      uint32_t* field1 = field0 + N*N*N + OFFSET/4;

      for(uint64_t i = 0; i < steps; ++i)
      {
        spinFlip(system0, system1, field0, seed++);
        spinFlip(system1, system0, field1, seed++);
      }

      // Rearrange system for output
      for(unsigned int z = 0; z < N; z++)
      for(unsigned int y = 0; y < N; y++)
      for(unsigned int x = 0; x < N; x++)
      {
        if((x+y+z)%2)
        {
          systemState[2*(z*N*N + y*N + x)]   = system0[z*N*N + y*N + x];
          systemState[2*(z*N*N + y*N + x)+1] = system1[z*N*N + y*N + x];
        }
        else
        {
          systemState[2*(z*N*N + y*N + x)]   = system1[z*N*N + y*N + x];
          systemState[2*(z*N*N + y*N + x)+1] = system0[z*N*N + y*N + x];
        }
      }
    }

    // Write out system
    mgr.SaveSystem(*it, reinterpret_cast<const char*>(systemState), system_size);

    // Statistics
    std::cout << "\r" << *it * 100.f/maxTime << "% finished" << std::flush;
  }

  double stopTime = get_cputime();
  double delta = stopTime-startTime;
  double timePerSpin = delta/64/N/N/N/maxTime;
  std::cout << "\r" << 100.0 << "% finished" << std::endl;
  std::cout << "Elapsed time: " << delta << std::endl;
  std::cout << "Time per spin: " << timePerSpin*1E12 << " ps" << std::endl;

  // Free memory
  #pragma offload target(mic:phi_card)       \
    out(systemState : length(2*N*N*N) FREE)  \
    nocopy(system0)
  {
    _mm_free(system0);
  }
}
