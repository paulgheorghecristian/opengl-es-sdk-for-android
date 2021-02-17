/* Copyright (c) 2013-2017, ARM Limited and Contributors
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

/**
 * \file Triangle.cpp
 * \brief A sample which shows how to draw a simple triangle to the screen.
 *
 * Uses a simple shader to fill the the triangle with a gradient color.
 */

#include <GLES2/gl2.h>
#include <GLES2/gl2ext.h>
 
#include <jni.h>
#include <android/log.h>

#include <cstdio>
#include <cstdlib>
#include <cmath> 

#include "AndroidPlatform.h"
#include "Triangle.h"
#include "Text.h"
#include "Shader.h"
#include "Timer.h"

#include <opencv2/core/mat.hpp>

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/photo/photo.hpp"


using std::string;
using namespace MaliSDK;

/* Asset directories and filenames. */
string resourceDirectory = "/data/data/com.arm.malideveloper.openglessdk.triangle/";
string vertexShaderFilename = "Triangle_triangle.vert";
string fragmentShaderFilename = "Triangle_triangle.frag";

string simpleVSFilename = "Simple.vert";
string simpleFSFilename = "Simple.frag";

string albedoFilename = "albedo.jpg";
string depthFilename = "depth.png";

/* Shader variables. */
GLuint programID;
GLint iLocPosition = -1;
GLint iLocUV = -1;

GLuint programID2;
GLint iLocPosition2 = -1;
GLint iLocUV2 = -1;

GLint iLocAlbedoTexture = -1;
GLint iLocDepthTexture = -1;
GLint iLocMaskTexture = -1;
GLint iLocProjectionMatrix = -1;
GLint iLocViewMatrix = -1;
GLint iLocModelMatrix = -1;

GLint iLocAlbedoTexture2 = -1;
GLint iLocProjectionMatrix2 = -1;
GLint iLocViewMatrix2 = -1;
GLint iLocModelMatrix2 = -1;


/* A text object to draw text on the screen. */ 
Text *text;

glm::mat4 projectionMatrix, viewMatrix, modelMatrix;

glm::vec3 cameraPosition, cameraRotation;
glm::vec3 position, rotation, scale;

float rotY = 0;

const static GLenum pixelFormat[5] = { 0, GL_LUMINANCE, GL_LUMINANCE_ALPHA, GL_RGB, GL_RGBA };
const static GLint internalFormat[5] = { 0, GL_LUMINANCE, GL_LUMINANCE_ALPHA, GL_RGB, GL_RGBA };

void init3DImageMesh(unsigned int width, unsigned int height);
bool initTexture();

void updateModelMatrix() {
    glm::mat4 T = glm::translate(glm::mat4(1.0), position);
    glm::mat4 R = glm::mat4_cast(glm::quat(glm::radians(rotation)));
    glm::mat4 S = glm::scale(glm::mat4(1.0), scale);

    modelMatrix = T*R*S;
}

void updateViewMatrix() {
    glm::quat xRotQuat = glm::angleAxis (-cameraRotation.x, glm::vec3(1,0,0));
    glm::quat yRotQuat = glm::angleAxis (-cameraRotation.y, glm::vec3(0,1,0));
    glm::quat zRotQuat = glm::angleAxis (-cameraRotation.z, glm::vec3(0,0,1));

    glm::quat orientation = xRotQuat * yRotQuat * zRotQuat;
    orientation = glm::normalize (orientation);

    viewMatrix = glm::mat4_cast (orientation);
    viewMatrix = glm::translate (viewMatrix, -cameraPosition);
}

bool setupGraphics(int width, int height)
{
    LOGD("setupGraphics(%d, %d)", width, height);

    /* Full paths to the shader files */
    string vertexShaderPath = resourceDirectory + vertexShaderFilename; 
    string fragmentShaderPath = resourceDirectory + fragmentShaderFilename;

    string vertexShaderPath2 = resourceDirectory + simpleVSFilename;
    string fragmentShaderPath2 = resourceDirectory + simpleFSFilename;

    GLuint vertexShaderID = 0;
    GLuint fragmentShaderID = 0;

    GLuint vertexShaderID2 = 0;
    GLuint fragmentShaderID2 = 0;

    GL_CHECK(glEnable(GL_DEPTH_TEST));
    GL_CHECK(glDepthFunc(GL_LEQUAL));
    
    
    /* Initialize OpenGL ES. */
    GL_CHECK(glEnable(GL_BLEND));
    /* Should do: src * (src alpha) + dest * (1-src alpha). */
    GL_CHECK(glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA));

    /* Initialize the Text object and add some text. */
    //text = new Text(resourceDirectory.c_str(), width, height);
    //text->addString(0, 0, "Simple triangle", 255, 255, 0, 255);

    /* Process shaders. */
    Shader::processShader(&vertexShaderID, vertexShaderPath.c_str(), GL_VERTEX_SHADER);
    LOGD("vertexShaderID = %d", vertexShaderID);
    Shader::processShader(&fragmentShaderID, fragmentShaderPath.c_str(), GL_FRAGMENT_SHADER);
    LOGD("fragmentShaderID = %d", fragmentShaderID);

    Shader::processShader(&vertexShaderID2, vertexShaderPath2.c_str(), GL_VERTEX_SHADER);
    LOGD("vertexShaderID = %d", vertexShaderID2);
    Shader::processShader(&fragmentShaderID2, fragmentShaderPath2.c_str(), GL_FRAGMENT_SHADER);
    LOGD("fragmentShaderID = %d", fragmentShaderID2);

    programID = GL_CHECK(glCreateProgram());
    if (programID == 0)
    {
        LOGE("Could not create program.");
        return false;
    }

    GL_CHECK(glAttachShader(programID, vertexShaderID));
    GL_CHECK(glAttachShader(programID, fragmentShaderID));
    GL_CHECK(glLinkProgram(programID));
    GL_CHECK(glUseProgram(programID));

    /* Positions. */
    GL_CHECK(iLocPosition = glGetAttribLocation(programID, "a_v3Position"));
    LOGD("glGetAttribLocation(\"a_v3Position\") = %d\n", iLocPosition);

    /* UV. */
    GL_CHECK(iLocUV = glGetAttribLocation(programID, "a_v2UV"));
    LOGD("glGetAttribLocation(\"a_v2UV\") = %d\n", iLocUV);

    iLocAlbedoTexture = GL_CHECK(glGetUniformLocation(programID, "u_AlbedoTexture"));
    GL_CHECK(glUniform1i(iLocAlbedoTexture, 0));
    iLocDepthTexture = GL_CHECK(glGetUniformLocation(programID, "u_DepthTexture"));
    GL_CHECK(glUniform1i(iLocDepthTexture, 1));
    iLocDepthTexture = GL_CHECK(glGetUniformLocation(programID, "u_DepthTexture"));
    GL_CHECK(glUniform1i(iLocDepthTexture, 1));
    iLocMaskTexture = GL_CHECK(glGetUniformLocation(programID, "u_MaskTexture"));
    GL_CHECK(glUniform1i(iLocMaskTexture, 2));

    iLocProjectionMatrix = GL_CHECK(glGetUniformLocation(programID, "projectionMatrix"));
    iLocViewMatrix = GL_CHECK(glGetUniformLocation(programID, "viewMatrix"));
    iLocModelMatrix = GL_CHECK(glGetUniformLocation(programID, "modelMatrix"));

    LOGD("iLocProjectionMatrix=%d", iLocProjectionMatrix);
    LOGD("iLocViewMatrix=%d", iLocViewMatrix);

    programID2 = GL_CHECK(glCreateProgram());
    if (programID2 == 0)
    {
        LOGE("Could not create program.");
        return false;
    }

    GL_CHECK(glAttachShader(programID2, vertexShaderID2));
    GL_CHECK(glAttachShader(programID2, fragmentShaderID2));
    GL_CHECK(glLinkProgram(programID2));
    GL_CHECK(glUseProgram(programID2));

    /* Positions. */
    GL_CHECK(iLocPosition2 = glGetAttribLocation(programID2, "a_v3Position"));
    LOGD("glGetAttribLocation(\"a_v3Position\") = %d\n", iLocPosition2);

    /* UV. */
    GL_CHECK(iLocUV2 = glGetAttribLocation(programID2, "a_v2UV"));
    LOGD("glGetAttribLocation(\"a_v2UV\") = %d\n", iLocUV2);

    iLocAlbedoTexture2 = GL_CHECK(glGetUniformLocation(programID2, "u_AlbedoTexture"));
    GL_CHECK(glUniform1i(iLocAlbedoTexture2, 0));

    iLocProjectionMatrix2 = GL_CHECK(glGetUniformLocation(programID2, "projectionMatrix"));
    iLocViewMatrix2 = GL_CHECK(glGetUniformLocation(programID2, "viewMatrix"));
    iLocModelMatrix2 = GL_CHECK(glGetUniformLocation(programID2, "modelMatrix"));

    GL_CHECK(glViewport(0, 0, width, height));

    init3DImageMesh(128, 128);

    projectionMatrix = glm::perspective(glm::radians(70.0f), (float) width / height, 1.0f, 1000.0f);
    cameraPosition = glm::vec3(0, 0, 50);
    cameraRotation = glm::vec3(0, 0, 0);
    updateViewMatrix();

    position = glm::vec3(0, 0, -60);
    rotation = glm::vec3(0);
    scale = glm::vec3(20, 40, 20);
    updateModelMatrix();

    GL_CHECK(glUseProgram(programID));
    glUniformMatrix4fv(iLocProjectionMatrix, 1, GL_FALSE, glm::value_ptr(projectionMatrix));
    glUniformMatrix4fv(iLocViewMatrix, 1, GL_FALSE, glm::value_ptr(viewMatrix));
    glUniformMatrix4fv(iLocModelMatrix, 1, GL_FALSE, glm::value_ptr(modelMatrix));

    GL_CHECK(glUseProgram(programID2));
    glUniformMatrix4fv(iLocProjectionMatrix2, 1, GL_FALSE, glm::value_ptr(projectionMatrix));
    glUniformMatrix4fv(iLocViewMatrix2, 1, GL_FALSE, glm::value_ptr(viewMatrix));
    glUniformMatrix4fv(iLocModelMatrix2, 1, GL_FALSE, glm::value_ptr(modelMatrix));

    initTexture();

    /* Set clear screen color. */
    GL_CHECK(glClearColor(1.0f, 1.0f, 1.0f, 1.0f));
    GL_CHECK(glClearDepthf(1.0f));
    glDisable(GL_CULL_FACE);

    return true;
}

void renderFrame(jfloat *gyroQuat)
{
    GL_CHECK(glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT));

    GL_CHECK(glUseProgram(programID));


    if (gyroQuat != NULL) {
        glm::quat gyroQuatGLM(gyroQuat[0], gyroQuat[1], gyroQuat[2], gyroQuat[3]);
        glm::vec3 euler = glm::eulerAngles(gyroQuatGLM);
        rotation = glm::vec3(0, rotY, 0);
        rotY -= euler.y * 0.8f;
        updateModelMatrix();
    }

    glUniformMatrix4fv(iLocProjectionMatrix, 1, GL_FALSE, glm::value_ptr(projectionMatrix));
    glUniformMatrix4fv(iLocViewMatrix, 1, GL_FALSE, glm::value_ptr(viewMatrix));
    glUniformMatrix4fv(iLocModelMatrix, 1, GL_FALSE, glm::value_ptr(modelMatrix));


    glBindBuffer(GL_ARRAY_BUFFER, vboHandles[VERTEX]);
    glEnableVertexAttribArray(iLocPosition);
    glVertexAttribPointer(iLocPosition, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void *) 0);
    glEnableVertexAttribArray(iLocUV);
    glVertexAttribPointer(iLocUV, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void *) (sizeof(glm::vec3)));

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vboHandles[INDEX]);

    GL_CHECK(glActiveTexture(GL_TEXTURE0));
    GL_CHECK(glBindTexture(GL_TEXTURE_2D, albedoTextureID));
    GL_CHECK(glActiveTexture(GL_TEXTURE1));
    GL_CHECK(glBindTexture(GL_TEXTURE_2D, depthTextureID));
    GL_CHECK(glActiveTexture(GL_TEXTURE2));
    GL_CHECK(glBindTexture(GL_TEXTURE_2D, maskTextureID));

    GL_CHECK(glDrawElements(GL_TRIANGLES, _3DImageMeshIndices.size(), GL_UNSIGNED_SHORT, (void *) 0));

    // BACKGROUND
    GL_CHECK(glUseProgram(programID2));

    glUniformMatrix4fv(iLocProjectionMatrix2, 1, GL_FALSE, glm::value_ptr(projectionMatrix));
    glUniformMatrix4fv(iLocViewMatrix2, 1, GL_FALSE, glm::value_ptr(viewMatrix));
    glUniformMatrix4fv(iLocModelMatrix2, 1, GL_FALSE, glm::value_ptr(modelMatrix));

    glBindBuffer(GL_ARRAY_BUFFER, vboBG);
    glEnableVertexAttribArray(iLocPosition2);
    glVertexAttribPointer(iLocPosition2, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void *) 0);
    glEnableVertexAttribArray(iLocUV2);
    glVertexAttribPointer(iLocUV2, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void *) (sizeof(glm::vec3)));

    GL_CHECK(glActiveTexture(GL_TEXTURE0));
    GL_CHECK(glBindTexture(GL_TEXTURE_2D, albedoBGTexture));

    GL_CHECK(glDrawArrays(GL_TRIANGLES, 0, _bgVertices.size()));

    /* Draw fonts. */
    //text->draw();
}

void init3DImageMesh(unsigned int numRectX, unsigned int numRectY) {
    const float RECT_WIDTH = 2.0f / numRectX;
    const float RECT_HEIGHT = 2.0f / numRectY;

    float currX;
    float currY = 1.0f;

    for (unsigned int y = 0; y < numRectY+1; y++) {
        currX = -1.0f;
        for (unsigned int x = 0; x < numRectX+1; x++) {
            Vertex vertex;

            vertex.position = glm::vec3(currX, currY, 0);
            vertex.UV = glm::vec2((currX + 1) * 0.5, (currY + 1) * 0.5);
            _3DImageMeshVertices.push_back(vertex);

            currX += RECT_WIDTH;
        }
        currY -= RECT_HEIGHT;
    }

    for (unsigned int y = 0; y < numRectY; y++) {
        for (unsigned int x = 0; x < numRectX; x++) {
            unsigned int start = x + y*(numRectY+1);
            _3DImageMeshIndices.push_back(start+0);
            _3DImageMeshIndices.push_back(start+1);
            _3DImageMeshIndices.push_back(start+numRectX+2);

            _3DImageMeshIndices.push_back(start+0);
            _3DImageMeshIndices.push_back(start+numRectX+2);
            _3DImageMeshIndices.push_back(start+numRectX+1);
        }
    }

    _bgVertices.push_back(Vertex(glm::vec3(-1.0f, 1.0f, 0.0f), glm::vec2(0, 1)));
    _bgVertices.push_back(Vertex(glm::vec3(-1.0f, -1.0f, 0.0f), glm::vec2(0, 0)));
    _bgVertices.push_back(Vertex(glm::vec3(1.0f, 1.0f, 0.0f), glm::vec2(1, 1)));

    _bgVertices.push_back(Vertex(glm::vec3(-1.0f, -1.0f, 0.0f), glm::vec2(0, 0)));
    _bgVertices.push_back(Vertex(glm::vec3(1.0f, -1.0f, 0.0f), glm::vec2(1, 0)));
    _bgVertices.push_back(Vertex(glm::vec3(1.0f, 1.0f, 0.0f), glm::vec2(1, 1)));
}

bool initTexture() {
    stbi_set_flip_vertically_on_load(true);

    string albedoFullFilename = resourceDirectory + albedoFilename;
    string depthFullFilename = resourceDirectory + depthFilename;

    int albedoWidth, albedoHeight, albedoChn;
    unsigned char *dataAlbedo = stbi_load(albedoFullFilename.c_str(), &albedoWidth, &albedoHeight, &albedoChn, 0);

    if (dataAlbedo == NULL) {
        LOGE("%s not found", albedoFullFilename.c_str());
        return false;
    }

    GL_CHECK(glGenTextures(1, &albedoTextureID));
    GL_CHECK(glBindTexture(GL_TEXTURE_2D, albedoTextureID));

    GL_CHECK(glTexImage2D(GL_TEXTURE_2D, 0, internalFormat[albedoChn],
                          albedoWidth, albedoHeight, 0,
                          pixelFormat[albedoChn], GL_UNSIGNED_BYTE, dataAlbedo));

    /* Set texture mode. */
    GL_CHECK(glGenerateMipmap(GL_TEXTURE_2D));
    GL_CHECK(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR));
    GL_CHECK(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)); /* Default anyway. */
    GL_CHECK(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT));
    GL_CHECK(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT));

    int width, height, chn;
    unsigned char *data = stbi_load(depthFullFilename.c_str(), &width, &height, &chn, 0);
    if (data == NULL) {
        LOGE("%s not found", depthFullFilename.c_str());
        return false;
    }

    cv::Mat src(height, width, CV_8UC1, data);
    cv::Mat dst, thresh;

    // sharpen image
    cv::bilateralFilter(src, dst, 15, 75, 75);

    // for contour detection
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::threshold(dst, thresh, 127, 255, cv::THRESH_BINARY);
    cv::findContours(thresh, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_NONE);

//    if (contours.size() > 0) {
//        for (std::size_t c = 0; c < hierarchy.size(); c++) {
//            if (hierarchy[c][3] == -1) {
//                // is root contour
//                std::vector<cv::Point> &contour = contours[c];
//                int width = thresh.cols;
//                int height = thresh.rows;
//
//                for (std::size_t v = 0; v < _3DImageMeshVertices.size(); v++) {
//                    // find point in mesh and remove
//                    for (cv::Point &p: contour) {
//                        glm::vec3 pos = _3DImageMeshVertices[v].position;
//
//                        float px = (((float) p.x / width) * 2.0f) - 1.0f;
//                        float py = (((float) p.y / height) * 2.0f) - 1.0f;
//
//                        glm::vec2 diff = glm::vec2(pos.x, pos.y) - glm::vec2(px, py);
//                        if (glm::length(diff) < 0.02f) {
//                            //_3DImageMeshVertices[v].position = glm::vec3(0);
//                            //_3DImageMeshVertices[v].UV = glm::vec2(-2, -2);
//                        }
//                    }
//                }
//            }
//        }
//    }

    glGenBuffers(NUM_VBOS, vboHandles);
    glBindBuffer(GL_ARRAY_BUFFER, vboHandles[VERTEX]);
    glBufferData(GL_ARRAY_BUFFER,
                 sizeof(Vertex) * _3DImageMeshVertices.size(),
                 &_3DImageMeshVertices[0],
                 GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vboHandles[INDEX]);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER,
                 sizeof(unsigned int) * _3DImageMeshIndices.size(),
                 &_3DImageMeshIndices[0],
                 GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

    GL_CHECK(glGenTextures(1, &depthTextureID));
    GL_CHECK(glBindTexture(GL_TEXTURE_2D, depthTextureID));

    GL_CHECK(glTexImage2D(GL_TEXTURE_2D, 0, internalFormat[chn], width, height, 0, pixelFormat[chn], GL_UNSIGNED_BYTE, dst.data));

    /* Set texture mode. */
    GL_CHECK(glGenerateMipmap(GL_TEXTURE_2D));
    GL_CHECK(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR));
    GL_CHECK(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)); /* Default anyway. */
    GL_CHECK(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT));
    GL_CHECK(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT));

    stbi_image_free(data);

    GL_CHECK(glGenTextures(1, &maskTextureID));
    GL_CHECK(glBindTexture(GL_TEXTURE_2D, maskTextureID));

    GL_CHECK(glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE, width, height, 0, GL_LUMINANCE, GL_UNSIGNED_BYTE, thresh.data));

    /* Set texture mode. */
    GL_CHECK(glGenerateMipmap(GL_TEXTURE_2D));
    GL_CHECK(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR));
    GL_CHECK(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)); /* Default anyway. */
    GL_CHECK(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT));
    GL_CHECK(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT));

    glGenBuffers(1, &vboBG);
    glBindBuffer(GL_ARRAY_BUFFER,
                 vboBG);
    glBufferData(GL_ARRAY_BUFFER,
                 sizeof(Vertex) * _bgVertices.size(),
                 &_bgVertices[0],
                 GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    cv::Mat albedoInpainted, threshResized;
    cv::Mat srcAlbedo(albedoHeight, albedoWidth, CV_8UC3, dataAlbedo);

    cv::resize(thresh, threshResized, cv::Size(albedoWidth, albedoHeight), 0, 0, cv::INTER_LINEAR);

    cv:inpaint(srcAlbedo, threshResized, albedoInpainted, 11, cv::INPAINT_TELEA);

    GL_CHECK(glGenTextures(1, &albedoBGTexture));
    GL_CHECK(glBindTexture(GL_TEXTURE_2D, albedoBGTexture));

    GL_CHECK(glTexImage2D(GL_TEXTURE_2D, 0, internalFormat[albedoChn],
                          albedoWidth, albedoHeight, 0,
                          pixelFormat[albedoChn], GL_UNSIGNED_BYTE, albedoInpainted.data));

    /* Set texture mode. */
    GL_CHECK(glGenerateMipmap(GL_TEXTURE_2D));
    GL_CHECK(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR));
    GL_CHECK(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)); /* Default anyway. */
    GL_CHECK(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT));
    GL_CHECK(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT));

    stbi_image_free(dataAlbedo);

    return true;
}

extern "C"
{
    JNIEXPORT void JNICALL Java_com_arm_malideveloper_openglessdk_triangle_Triangle_init
    (JNIEnv *env, jclass jcls, jint width, jint height)
    {
        /* Make sure that all resource files are in place. */
        AndroidPlatform::getAndroidAsset(env, resourceDirectory.c_str(), vertexShaderFilename.c_str());
        AndroidPlatform::getAndroidAsset(env, resourceDirectory.c_str(), fragmentShaderFilename.c_str());

        AndroidPlatform::getAndroidAsset(env, resourceDirectory.c_str(), simpleVSFilename.c_str());
        AndroidPlatform::getAndroidAsset(env, resourceDirectory.c_str(), simpleFSFilename.c_str());

        AndroidPlatform::getAndroidAsset(env, resourceDirectory.c_str(), albedoFilename.c_str());
        AndroidPlatform::getAndroidAsset(env, resourceDirectory.c_str(), depthFilename.c_str());

        setupGraphics(width, height);
    }

    JNIEXPORT void JNICALL Java_com_arm_malideveloper_openglessdk_triangle_Triangle_step
    (JNIEnv *env, jclass jcls, jfloatArray gyroQuat)
    {
        jfloat * quadData = env->GetFloatArrayElements(gyroQuat, 0);
        renderFrame(quadData);
    }

    JNIEXPORT void JNICALL Java_com_arm_malideveloper_openglessdk_triangle_Triangle_uninit
    (JNIEnv *, jclass)
    {
        delete text;
        text = NULL;
    }
}
