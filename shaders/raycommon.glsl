// shared ray tracing struct
struct hitPayload
{
  vec3 hitValue;
  uint seed;
  uint depth;
  vec3 rayOrigin;
  vec3 rayDirection;
  vec3 weight;
  vec3 hitPos;
  vec3 hitNorm;
};

//storing norm as single uint
const float C_Stack_Max = 3.402823466e+38f;

uint CompressUnitVec(vec3 nv)
{
  // map to octahedron and then flatten to 2D (see 'Octahedron Environment Maps' by Engelhardt & Dachsbacher)
  if((nv.x < C_Stack_Max) && !isinf(nv.x))
  {
    float d = 32767.0f / (abs(nv.x) + abs(nv.y) + abs(nv.z));
    int         x = int(roundEven(nv.x * d));
    int         y = int(roundEven(nv.y * d));
    if(nv.z < 0.0f)
    {
      int maskx = x >> 31;
      int masky = y >> 31;
      int tmp   = 32767 + maskx + masky;
      int tmpx  = x;
      x               = (tmp - (y ^ masky)) ^ maskx;
      y               = (tmp - (tmpx ^ maskx)) ^ masky;
    }
    uint packed = (uint(y + 32767) << 16) | uint(x + 32767);
    if(packed == ~0u)
      return ~0x1u;
    return packed;
  }
  else
  {
    return ~0u;
  }
};

float ShortToFloatM11(int v)  // linearly maps a short 32767-32768 to a float -1-+1
{
  return (v >= 0) ? (uintBitsToFloat(0x3F800000u | (uint(v) << 8)) - 1.0f) :
                    (uintBitsToFloat((0x80000000u | 0x3F800000u) | (uint(-v) << 8)) + 1.0f);
};

vec3 DecompressUnitVec(uint packed)
{
  if(packed != ~0u)  // sanity check, not needed as isvalid_unit_vec is called earlier
  {
    int       x     = int(packed & 0xFFFFu) - 32767;
    int       y     = int(packed >> 16) - 32767;
    int maskx = x >> 31;
    int masky = y >> 31;
    int tmp0  = 32767 + maskx + masky;
    int ymask = y ^ masky;
    int tmp1  = tmp0 - (x ^ maskx);
    int z     = tmp1 - ymask;
    float     zf;
    if(z < 0)
    {
      x  = (tmp0 - ymask) ^ maskx;
      y  = tmp1 ^ masky;
      zf = uintBitsToFloat((0x80000000u | 0x3F800000u) | (uint(-z) << 8)) + 1.0f;
    }
    else
    {
      zf = uintBitsToFloat(0x3F800000u | (uint(z) << 8)) - 1.0f;
    }
    return normalize(vec3(ShortToFloatM11(x), ShortToFloatM11(y), zf));
  }
  else
  {
    return vec3(C_Stack_Max);
  }
};