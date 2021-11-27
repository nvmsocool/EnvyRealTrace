/*
 * Copyright (c) 2014-2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-FileCopyrightText: Copyright (c) 2014-2021 NVIDIA CORPORATION
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "nvvk/appbase_vk.hpp"
#include "nvvk/debug_util_vk.hpp"
#include "nvvk/descriptorsets_vk.hpp"
#include "nvvk/memallocator_dma_vk.hpp"
#include "nvvk/resourceallocator_vk.hpp"
#include "shaders/host_device.h"
#include "nvvk/raytraceKHR_vk.hpp"

//--------------------------------------------------------------------------------------------------
// Simple rasterizer of OBJ objects
// - Each OBJ loaded are stored in an `ObjModel` and referenced by a `ObjInstance`
// - It is possible to have many `ObjInstance` referencing the same `ObjModel`
// - Rendering is done in an offscreen framebuffer
// - The image of the framebuffer is displayed in post-process in a full-screen quad
//
class HelloVulkan : public nvvk::AppBaseVk
{
public:
  void   setup(const VkInstance& instance, const VkDevice& device, const VkPhysicalDevice& physicalDevice, uint32_t queueFamily) override;
  void   createDescriptorSetLayout();
  void   createGraphicsPipeline();
  size_t loadModel(const std::string& filename, nvmath::mat4f transform = nvmath::mat4f(1));
  void   updateDescriptorSet();
  void   createUniformBuffer();
  void   createObjDescriptionBuffer();
  void   createTextureImages(const VkCommandBuffer& cmdBuf, const std::vector<std::string>& textures);
  void   updateUniformBuffer(const VkCommandBuffer& cmdBuf);
  void   onResize(int /*w*/, int /*h*/) override;
  void   destroyResources();
  void   rasterize(const VkCommandBuffer& cmdBuff);

  // The OBJ model
  struct ObjModel
  {
    uint32_t     nbIndices{0};
    uint32_t     nbVertices{0};
    nvvk::Buffer vertexBuffer;    // Device buffer of all 'Vertex'
    nvvk::Buffer indexBuffer;     // Device buffer of the indices forming triangles
    nvvk::Buffer matColorBuffer;  // Device buffer of array of 'Wavefront material'
    nvvk::Buffer matIndexBuffer;  // Device buffer of array of 'Wavefront material'
    float        max_lum = 0;
  };

  struct ObjInstance
  {
    nvmath::mat4f transform;    // Matrix of the instance
    uint32_t      objIndex{0};  // Model index reference
  };


  // Information pushed at each draw call
  PushConstantRaster m_pcRaster{
      {1},                // Identity matrix
      {1.5f, 3.5f, 0.f},  // light position
      0,                  // instance Id
      1.f,                // light intensity
      0                   // light type
  };

  struct rt_light
  {
    float area, probability;
    vec3  p1, p2, p3;
    vec3  p12, p13;
  };

  // Array of objects and instances in the scene
  std::vector<ObjModel>    m_objModel;   // Model on host
  std::vector<ObjDesc>     m_objDesc;    // Model description for device access
  std::vector<ObjInstance> m_instances;  // Scene model instances
  std::vector<rt_light>    m_lights;


  // Graphic pipeline
  VkPipelineLayout            m_pipelineLayout;
  VkPipeline                  m_graphicsPipeline;
  nvvk::DescriptorSetBindings m_descSetLayoutBind;
  VkDescriptorPool            m_descPool;
  VkDescriptorSetLayout       m_descSetLayout;
  VkDescriptorSet             m_descSet;

  nvvk::Buffer m_bGlobals;  // Device-Host of the camera matrices
  nvvk::Buffer m_bObjDesc;  // Device buffer of the OBJ descriptions

  std::vector<nvvk::Texture> m_textures;  // vector of all textures of the scene


  nvvk::ResourceAllocatorDma m_alloc;  // Allocator for buffer, images, acceleration structures
  nvvk::DebugUtil            m_debug;  // Utility to name objects


  // #Post - Draw the rendered image on a quad using a tonemapper
  void createOffscreenRender();
  void createPostPipeline();
  void createPostDescriptor();
  void updatePostDescriptorSet();
  void drawPost(VkCommandBuffer cmdBuf);

  nvvk::DescriptorSetBindings m_postDescSetLayoutBind;
  VkDescriptorPool            m_postDescPool{VK_NULL_HANDLE};
  VkDescriptorSetLayout       m_postDescSetLayout{VK_NULL_HANDLE};
  VkDescriptorSet             m_postDescSet{VK_NULL_HANDLE};
  VkPipeline                  m_postPipeline{VK_NULL_HANDLE};
  VkPipelineLayout            m_postPipelineLayout{VK_NULL_HANDLE};
  VkRenderPass                m_offscreenRenderPass{VK_NULL_HANDLE};
  VkFramebuffer               m_offscreenFramebuffer{VK_NULL_HANDLE};
  nvvk::Texture               m_gBuffer, m_gBufferHistory;
  nvvk::Texture               m_rtCurrentBuffer, m_posCurrentBuffer, m_rtHistoryBuffer, m_posHistoryBuffer;
  nvvk::Texture m_iterationCurrentBuffer, m_normalCurrentBuffer, m_iterationHistoryBuffer, m_normalHistoryBuffer;
  nvvk::Texture m_denoiseBuffer;
  nvvk::Texture m_outputImageBuffer;
  VkFormat      m_offscreenColorFormat{VK_FORMAT_R32G32B32A32_SFLOAT};
  VkFormat      m_offscreenDepthFormat{VK_FORMAT_X8_D24_UNORM_PACK32};

  // #VKRay
  void                                            initRayTracing();
  VkPhysicalDeviceRayTracingPipelinePropertiesKHR m_rtProperties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR};

  // Accelleration structure objects and functions
  nvvk::RaytracingBuilderKHR m_rtBuilder;
  auto                       objectToVkGeometryKHR(const ObjModel& model);
  void                       createBottomLevelAS();
  void                       createTopLevelAS();

  //Descriptor objects and functions
  nvvk::DescriptorSetBindings m_rtDescSetLayoutBind;
  VkDescriptorPool            m_rtDescPool;
  VkDescriptorSetLayout       m_rtDescSetLayout;
  VkDescriptorSet             m_rtDescSet;

  void createRtDescriptorSet();
  void updateRtDescriptorSet();

  //rt pipeline
  void createRtPipeline();

  std::vector<VkRayTracingShaderGroupCreateInfoKHR> m_rtShaderGroups;
  VkPipelineLayout                                  m_rtPipelineLayout;
  VkPipeline                                        m_rtPipeline;

  // Push constant for ray tracer
  PushConstantRay m_pcRay{};


  //binding table
  void createRtShaderBindingTable();

  nvvk::Buffer                    m_rtSBTBuffer;
  VkStridedDeviceAddressRegionKHR m_rgenRegion{};
  VkStridedDeviceAddressRegionKHR m_missRegion{};
  VkStridedDeviceAddressRegionKHR m_hitRegion{};
  VkStridedDeviceAddressRegionKHR m_callRegion{};

  // function to add rt call to commnd buffer
  void raytrace(const VkCommandBuffer& cmdBuf, const nvmath::vec4f& clearColor);

  //AA
  void ResetFrame();
  void updateFrame();

  //history
  mat4        m_priorViewProj;
  VkImageCopy m_copy_region;

  float m_maxAnis = 0;

  // denoiser compute pass setup
  void createDenoiseDescriptorSet();
  void createDenoiseCompPipeline();
  void updateDenoiseCompDescriptors();
  void runCompute(VkCommandBuffer cmdBuf);

  nvvk::DescriptorSetBindings m_denoiseDescSetLayoutBind;
  VkDescriptorPool            m_denoiseDescPool;
  VkDescriptorSetLayout       m_denoiseDescSetLayout;
  VkDescriptorSet             m_denoiseDescSet;
  VkPipelineLayout            m_denoiseCompPipelineLayout;
  VkPipeline                  m_denoisePipelineX, m_denoisePipelineY;
  PushConstantDenoise         m_denoisePushConstants;
  int                         m_num_atrous_iterations = 0;

  std::vector<std::vector<float>> m_blue_noise;
  std::vector<uint>               m_flat_blue_noise;
  void                            populate_blue_noise();
  float                           get_local_avg(size_t i, size_t j);
  nvvk::Texture                   m_blueNoiseBuffer;
};
