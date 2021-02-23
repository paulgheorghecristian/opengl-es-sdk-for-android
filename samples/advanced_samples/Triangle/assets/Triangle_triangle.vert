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

attribute vec3 a_v3Position;
attribute vec2 a_v2UV;

varying vec2 v_v2UV;

uniform mat4 projectionMatrix, viewMatrix, modelMatrix;

uniform sampler2D u_DepthTexture;

void main()
{
    v_v2UV = vec2(a_v2UV.x, 1.0 - a_v2UV.y);
    vec4 v4Texel = texture2D(u_DepthTexture, v_v2UV);

    //gl_Position = projectionMatrix * viewMatrix * modelMatrix * vec4(a_v3Position.xy, 0.5, 1.0);
    gl_Position = projectionMatrix * viewMatrix * modelMatrix * vec4(a_v3Position.xy, v4Texel.x, 1.0);
    //gl_Position = vec4(a_v3Position.xy, 1.0, 1.0);
}