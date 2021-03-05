/* Copyright (c) 2012-2017, ARM Limited and Contributors
 *
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge,
 * to any person obtaining a copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation the rights to
 * use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
 * and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
 * WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#ifndef TRIANGLE_H
#define TRIANGLE_H

#define GLES_VERSION 2

#define GLM_FORCE_RADIANS

#include <GLES2/gl2.h>
#include <glm/glm.hpp>
#include <vector>

#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

using uint = unsigned int;
using uchar = unsigned char;

#include <stb/stb_image.h>
#include <stb/stb_image_write.h>


/* Simple triangle. */
const float triangleVertices[] =
{
     0.0f,  0.5f, 0.0f,
    -0.5f, -0.5f, 0.0f,
     0.5f, -0.5f, 0.0f,
};

/* Per corner colors for the triangle (Red, Green, Green). */
const float triangleColors[] =
{
    1.0, 0.0, 0.0, 1.0,
    0.0, 1.0, 0.0, 1.0,
    0.0, 1.0, 0.0, 1.0,
};

struct Vertex{
    glm::vec3 position;
    glm::vec2 UV;

    Vertex(const glm::vec3 &pos, const glm::vec2 &UV): position(pos), UV(UV) {}
    Vertex(): position(0), UV(0) {}
};

std::vector<Vertex> _3DImageMeshVertices;
std::vector<unsigned short> _3DImageMeshIndices;

std::vector<Vertex> _bgVertices;
std::vector<unsigned short> _bgIndices;

enum VBOs{
    VERTEX = 0, INDEX, NUM_VBOS
};
GLuint vboHandles[NUM_VBOS];
GLuint albedoTextureID = 0, depthTextureID = 0, maskTextureID = 0, depthTextureBGID = 0;

GLuint vboBG[NUM_VBOS];
GLuint albedoBGTexture = 0;

#endif /* TRIANGLE_H */