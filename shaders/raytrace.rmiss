#version 460
#extension GL_EXT_ray_tracing : require

#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

#include "raycommon.glsl"
#include "wavefront.glsl"

layout(location = 0) rayPayloadInEXT hitPayload prd;

layout(push_constant) uniform _PushConstantRay
{
  PushConstantRay pcRay;
};

void main()
{
  if(prd.depth == 0)
  {
    prd.hitValue = vec3(.1,.1,.1) * 0.8;
  }
  else
  {
    prd.hitValue = vec3(0.0);  // Tiny contribution from environment
  }
  prd.depth = 100;              // Ending trace
}