#version 460
#extension GL_EXT_shader_explicit_arithmetic_types_int64  : require
#extension GL_GOOGLE_include_directive : enable

#include "host_device.h"

layout(location = 0) in vec2 outUV;
layout(location = 0) out vec4 fragColor;

layout(binding = 0, set = 0, rgba32f) uniform image2D denoised;
layout(binding = 1, set = 0, rgba32f) uniform image2D priorFrame;
layout(binding = 2, set = 0) uniform sampler2D posHistory;
layout(binding = 3, set = 0, rgba32f) uniform image2D pos;

// layout(push_constant) uniform _PushConstantPost { PushConstantPost pcPost; };

void main()
{
    // ivec2 imgUV = ivec2(pcPost.w*outUV.x,pcPost.h*outUV.y);
    // 
    // //get this pixel's location
    // vec3 firstHitPos = imageLoad(pos, imgUV).xyz;
    // 
    // // denoised color from rt+denoise
    // vec3 hitValue = imageLoad(denoised, imgUV).xyz;
    // 
    // //return;
    // 
    // // find equivelant fragment from last frame, based on world position
    // // store camera perpectice from last frame, re-calculate screen space of point
    // 
    // vec4 screenProjected = (pcPost.priorViewProj * vec4(firstHitPos, 1.0));
    // vec2 screenspace = ((screenProjected.xy / screenProjected.w) + vec2(1.0)) / 2.0;
    // 
    // float frame = 0;
    // vec3 old_color;
    // 
    // if (!(screenspace.x < 0 || screenspace.x > 1 || screenspace.y < 0 || screenspace.y > 1))
    // {
    //   // fragment was visible in last camera view
    //   vec3 priorFirstHit = texture(posHistory, screenspace).xyz;
    // 
    //   if (length(priorFirstHit - firstHitPos) < pcPost.posTolerance)
    //   {
    //     // fragment was not occluded in last camera view
    //     // pull color and frame from colorHistory
    //     //vec4 history = texture(colorHistory, screenspace);
    //     vec2 loc_f = screenspace * ivec2(pcPost.w,pcPost.h);
    //     ivec2 location = ivec2(floor(loc_f));
    //     vec4 history = imageLoad(priorFrame, location);
    //     old_color = history.xyz;
    // 
    //     //map 0,1, with 1 = max error, to 1 * subPixelReduction, 0
    //     
    //     frame = history.w;
    //   }
    // }
    // 
    // if(frame >= 1.0 && frame > 0)
    // {
    //   float a         = 1.0f / frame;
    //   vec3  new_color = vec3(mix(old_color, hitValue, a));
    //   fragColor = vec4(new_color, frame+1);
    // }
    // else
    // {
    //   // First frame, replace the value in the buffer
    //   fragColor = vec4(hitValue, 1);
    // }
    
  vec2  uv    = outUV;
  float gamma = 1. / 2.2;
  fragColor   = vec4(pow(fragColor.xyz, vec3(gamma)),fragColor.w);
}
