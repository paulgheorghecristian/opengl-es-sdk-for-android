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
#include <opencv2/core.hpp>

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/photo/photo.hpp"
#include <opencv2/dnn/dnn.hpp>

//#include <caffe2/predictor/predictor.h>
//#include <caffe2/core/operator.h>
//#include <caffe2/core/timer.h>
//#include <caffe2/core/tensor.h>
//#include "caffe2/core/init.h"
#include <omp.h>

#include <torch/script.h>

#include <fstream>

using std::string;
using namespace MaliSDK;

/* Asset directories and filenames. */
string resourceDirectory = "/data/data/com.arm.malideveloper.openglessdk.triangle/";
string vertexShaderFilename = "3DMesh.vert";
string fragmentShaderFilename = "3DMesh.frag";

string simpleVSFilename = "bg.vert";
string simpleFSFilename = "bg.frag";

string albedoFilename = "photo.jpg";
string depthFilename = "depth.png";

string predictNet = "tiefenrausch.pb";
string initNet = "tiefenrausch_init.pb";

string predictNet2 = "u2netp_predict.pb";
string initNet2 = "u2netp_init.pb";

string modelPath = "midas_torchscript.pt";
string u2netp_modelPath = "u2netp_torchscript.pt";

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
GLint iLocDepthTexture2 = -1;
GLint iLocMaskTexture2 = -1;
GLint iLocProjectionMatrix2 = -1;
GLint iLocViewMatrix2 = -1;
GLint iLocModelMatrix2 = -1;


/* A text object to draw text on the screen. */ 
Text *text;

glm::mat4 projectionMatrix, viewMatrix, modelMatrix;
glm::mat4 modelMatrix2;

glm::vec3 cameraPosition, cameraRotation;
glm::vec3 position, rotation, scale;

glm::vec3 position2, rotation2, scale2;

static torch::jit::Module module, module2;

float rotY = 0, rotX = 0;

const static GLenum pixelFormat[5] = { 0, GL_LUMINANCE, GL_LUMINANCE_ALPHA, GL_RGB, GL_RGBA };
const static GLint internalFormat[5] = { 0, GL_LUMINANCE, GL_LUMINANCE_ALPHA, GL_RGB, GL_RGBA };

//static caffe2::NetDef _initNet, _predictNet;
//static caffe2::Predictor *_predictor = NULL;
//
//static caffe2::NetDef _initNet2, _predictNet2;
//static caffe2::Predictor *_predictor2 = NULL;

static const int SALIENCE_DIM = 320;


void init3DImageMesh(unsigned int width, unsigned int height);
bool initTexture();
void initTorch();
bool inferDepth(unsigned char *dataAlbedo,
                int albedoWidth,
                int albedoHeight,
                torch::Tensor &t_out);
bool inferSalience(unsigned char *dataAlbedo,
                    int albedoWidth,
                    int albedoHeight,
                    torch::jit::IValue &t_out);
void fillHoles(cv::Mat &in, cv::Mat &out);

float heightNorm = 0.0f;
glm::vec3 pointOfRotation;

struct JITCallGuard {
    torch::autograd::AutoGradMode no_autograd_guard{false};
    torch::AutoNonVariableTypeMode non_var_guard{true};
    torch::jit::GraphOptimizerEnabledGuard no_optimizer_guard{false};
};

static int compare(const void* a, const void* b)
{
    const unsigned char* x = (unsigned char*) a;
    const unsigned char* y = (unsigned char*) b;

    if (*x > *y)
        return 1;
    else if (*x < *y)
        return -1;

    return 0;
}

void initCaffe2() {
//    bool rs = caffe2::GlobalInit();
//
//    CAFFE_ENFORCE(caffe2::ReadProtoFromFile(resourceDirectory+initNet, &_initNet));
//    CAFFE_ENFORCE(caffe2::ReadProtoFromFile(resourceDirectory+predictNet, &_predictNet));
//
//    _predictNet.mutable_device_option()->set_device_type((int) caffe2::CPU);
//    _initNet.mutable_device_option()->set_device_type((int) caffe2::CPU);
//    for(int i = 0; i < _predictNet.op_size(); ++i){
//        _predictNet.mutable_op(i)->mutable_device_option()->set_device_type((int) caffe2::CPU);
//    }
//    for(int i = 0; i < _initNet.op_size(); ++i){
//        _initNet.mutable_op(i)->mutable_device_option()->set_device_type((int) caffe2::CPU);
//    }
//
//    _predictor = new caffe2::Predictor(_initNet, _predictNet);
//
//    CAFFE_ENFORCE(caffe2::ReadProtoFromFile(resourceDirectory+initNet2, &_initNet2));
//    CAFFE_ENFORCE(caffe2::ReadProtoFromFile(resourceDirectory+predictNet2, &_predictNet2));
//
//    _predictNet2.mutable_device_option()->set_device_type((int) caffe2::CPU);
//    _initNet2.mutable_device_option()->set_device_type((int) caffe2::CPU);
//    for(int i = 0; i < _predictNet2.op_size(); ++i){
//        _predictNet2.mutable_op(i)->mutable_device_option()->set_device_type((int) caffe2::CPU);
//    }
//    for(int i = 0; i < _initNet2.op_size(); ++i){
//        _initNet2.mutable_op(i)->mutable_device_option()->set_device_type((int) caffe2::CPU);
//    }
//
//    _predictor2 = new caffe2::Predictor(_initNet2, _predictNet2);
}

void initTorch() {
    JITCallGuard guard;
    module = torch::jit::load(resourceDirectory+modelPath);
    module.eval();

    module2 = torch::jit::load(resourceDirectory+u2netp_modelPath);
    module2.eval();
}

void updateModelMatrix() {
    glm::mat4 T = glm::translate(glm::mat4(1.0), position);
    glm::mat4 R = glm::mat4_cast(glm::quat(glm::radians(rotation)));
    glm::mat4 S = glm::scale(glm::mat4(1.0), scale);

    glm::mat4 Tc = glm::translate(glm::mat4(1.0), -pointOfRotation);
    glm::mat4 Tcp = glm::translate(glm::mat4(1.0), pointOfRotation);

    modelMatrix = T*Tcp*R*Tc*S;

    glm::mat4 T2 = glm::translate(glm::mat4(1.0), position2);
    glm::mat4 R2 = glm::mat4_cast(glm::quat(glm::radians(rotation2)));
    glm::mat4 S2 = glm::scale(glm::mat4(1.0), scale2);

    modelMatrix2 = T2*R2*S2;
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
    iLocDepthTexture2 = GL_CHECK(glGetUniformLocation(programID2, "u_DepthTexture"));
    GL_CHECK(glUniform1i(iLocDepthTexture2, 1));
    iLocMaskTexture2 = GL_CHECK(glGetUniformLocation(programID2, "u_MaskTexture"));
    GL_CHECK(glUniform1i(iLocMaskTexture2, 2));

    iLocProjectionMatrix2 = GL_CHECK(glGetUniformLocation(programID2, "projectionMatrix"));
    iLocViewMatrix2 = GL_CHECK(glGetUniformLocation(programID2, "viewMatrix"));
    iLocModelMatrix2 = GL_CHECK(glGetUniformLocation(programID2, "modelMatrix"));

    GL_CHECK(glViewport(0, 0, width, height));

    init3DImageMesh(128, 128);

    projectionMatrix = glm::perspective(glm::radians(70.0f), (float) width / height, 1.0f, 5000.0f);
    cameraPosition = glm::vec3(0, 0, 60);
    cameraRotation = glm::vec3(0, 0, 0);
    updateViewMatrix();

    position = glm::vec3(0, 0, -24);
    rotation = glm::vec3(10.0f, 0, 0);
    scale = glm::vec3(20, 40, 30);
    position2 = glm::vec3(0, 0, -30);
    rotation2 = glm::vec3(0.0f, 0, 0);
    scale2 = glm::vec3(20, 40, 30);
    updateModelMatrix();

    GL_CHECK(glUseProgram(programID));
    glUniformMatrix4fv(iLocProjectionMatrix, 1, GL_FALSE, glm::value_ptr(projectionMatrix));
    glUniformMatrix4fv(iLocViewMatrix, 1, GL_FALSE, glm::value_ptr(viewMatrix));
    glUniformMatrix4fv(iLocModelMatrix, 1, GL_FALSE, glm::value_ptr(modelMatrix));

    GL_CHECK(glUseProgram(programID2));
    glUniformMatrix4fv(iLocProjectionMatrix2, 1, GL_FALSE, glm::value_ptr(projectionMatrix));
    glUniformMatrix4fv(iLocViewMatrix2, 1, GL_FALSE, glm::value_ptr(viewMatrix));
    glUniformMatrix4fv(iLocModelMatrix2, 1, GL_FALSE, glm::value_ptr(modelMatrix2));

    initTexture();

    /* Set clear screen color. */
    GL_CHECK(glClearColor(1.0f, 1.0f, 1.0f, 1.0f));
    GL_CHECK(glClearDepthf(1.0f));
    glDisable(GL_CULL_FACE);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    return true;
}

void renderFrame(jfloat *gyroQuat)
{
    GL_CHECK(glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT));

    if (gyroQuat != NULL) {
        glm::quat gyroQuatGLM(gyroQuat[0], gyroQuat[1], gyroQuat[2], gyroQuat[3]);
        glm::vec3 euler = glm::eulerAngles(gyroQuatGLM);
        //cameraRotation = glm::vec3(0.1f*glm::cos(rotY), 0.1f*glm::sin(rotY), 0);
        //rotation = glm::vec3(5.0f*glm::cos(rotY), 5.0f*glm::sin(rotY), 0);
        //rotation2 = glm::vec3(1.0f*glm::cos(rotY), 1.0f*glm::sin(rotY), 0);
        cameraPosition = glm::vec3(10.0f*glm::cos(rotY), 8.0f*glm::sin(rotY), 60);
        cameraRotation = glm::vec3(0, rotY, rotY)*0.02f;
        rotY += 0.03f;
        updateModelMatrix();
        updateViewMatrix();
        viewMatrix = glm::lookAt(cameraPosition, glm::vec3(0.0f), glm::vec3(0, 1, 0));
    }

    // BACKGROUND
    GL_CHECK(glUseProgram(programID2));

    glUniformMatrix4fv(iLocProjectionMatrix2, 1, GL_FALSE, glm::value_ptr(projectionMatrix));
    glUniformMatrix4fv(iLocViewMatrix2, 1, GL_FALSE, glm::value_ptr(viewMatrix));
    glUniformMatrix4fv(iLocModelMatrix2, 1, GL_FALSE, glm::value_ptr(modelMatrix2));

    glBindBuffer(GL_ARRAY_BUFFER, vboBG[VERTEX]);
    glEnableVertexAttribArray(iLocPosition2);
    glVertexAttribPointer(iLocPosition2, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void *) 0);
    glEnableVertexAttribArray(iLocUV2);
    glVertexAttribPointer(iLocUV2, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void *) (sizeof(glm::vec3)));

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vboBG[INDEX]);

    GL_CHECK(glActiveTexture(GL_TEXTURE0));
    GL_CHECK(glBindTexture(GL_TEXTURE_2D, albedoBGTexture));
    GL_CHECK(glActiveTexture(GL_TEXTURE1));
    GL_CHECK(glBindTexture(GL_TEXTURE_2D, depthTextureBGID));
    GL_CHECK(glActiveTexture(GL_TEXTURE2));
    GL_CHECK(glBindTexture(GL_TEXTURE_2D, maskTextureID));

    GL_CHECK(glDrawElements(GL_TRIANGLES, _bgIndices.size(), GL_UNSIGNED_SHORT, (void *) 0));


    GL_CHECK(glUseProgram(programID));

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
            _bgVertices.push_back(vertex);

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

            _bgIndices.push_back(start+0);
            _bgIndices.push_back(start+1);
            _bgIndices.push_back(start+numRectX+2);

            _bgIndices.push_back(start+0);
            _bgIndices.push_back(start+numRectX+2);
            _bgIndices.push_back(start+numRectX+1);
        }
    }

//    _bgVertices.push_back(Vertex(glm::vec3(-1.0f, 1.0f, 0.0f), glm::vec2(0, 1)));
//    _bgVertices.push_back(Vertex(glm::vec3(-1.0f, -1.0f, 0.0f), glm::vec2(0, 0)));
//    _bgVertices.push_back(Vertex(glm::vec3(1.0f, 1.0f, 0.0f), glm::vec2(1, 1)));
//
//    _bgVertices.push_back(Vertex(glm::vec3(-1.0f, -1.0f, 0.0f), glm::vec2(0, 0)));
//    _bgVertices.push_back(Vertex(glm::vec3(1.0f, -1.0f, 0.0f), glm::vec2(1, 0)));
//    _bgVertices.push_back(Vertex(glm::vec3(1.0f, 1.0f, 0.0f), glm::vec2(1, 1)));
}

bool inferDepth(unsigned char *dataAlbedo,
                int albedoWidth,
                int albedoHeight,
                float *salienceData,
                torch::Tensor &t_out) {
    int currW = albedoWidth;
    int currH = albedoHeight;
    cv::Mat srcAlbedo(albedoHeight, albedoWidth, CV_8UC4, dataAlbedo);
    cv::Mat srcAlbedoRGB, albedoResized;

    float aspectRatio = (float) albedoWidth / albedoHeight;

    const int targetMaxDimension = 384;
    if (albedoHeight > albedoWidth) {
        albedoHeight = targetMaxDimension;
        albedoWidth = (int) floor(albedoHeight * aspectRatio);
    } else {
        albedoWidth = targetMaxDimension;
        albedoHeight = (int) floor(albedoWidth / aspectRatio);
    }
    albedoWidth -= albedoWidth % 32;
    albedoHeight -= albedoHeight % 32;

    albedoWidth = 320;
    albedoHeight = 320;

    cv::cvtColor(srcAlbedo, srcAlbedoRGB, cv::COLOR_RGBA2RGB);
    cv::resize(srcAlbedoRGB, albedoResized, cv::Size(albedoWidth, albedoHeight), 0, 0, cv::INTER_AREA);

    float *dataCHW = new float[3*320*320];
    for (std::size_t i = 0; i < albedoHeight; i++) {
        for (std::size_t j = 0; j < albedoWidth; j++) {
            dataCHW[0 * albedoHeight * albedoWidth + albedoWidth*i + j] =
                    (float) (albedoResized.data[3*(i * albedoWidth + j) + 0]) / 255.0f;
            dataCHW[0 * albedoHeight * albedoWidth + albedoWidth*i + j] =
                    (dataCHW[0 * albedoHeight * albedoWidth + albedoWidth*i + j] - 0.485f) / 0.229f;

            dataCHW[1 * albedoHeight * albedoWidth + albedoWidth*i + j] =
                    (float) (albedoResized.data[3*(i * albedoWidth + j) + 1]) / 255.0f;
            dataCHW[1 * albedoHeight * albedoWidth + albedoWidth*i + j] =
                    (dataCHW[1 * albedoHeight * albedoWidth + albedoWidth*i + j] - 0.456f) / 0.224f;

            dataCHW[2 * albedoHeight * albedoWidth + albedoWidth*i + j] =
                    (float) (albedoResized.data[3*(i * albedoWidth + j) + 2]) / 255.0f;
            dataCHW[2 * albedoHeight * albedoWidth + albedoWidth*i + j] =
                    (dataCHW[2 * albedoHeight * albedoWidth + albedoWidth*i + j] - 0.406f) / 0.225f;
        }
    }

    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
    torch::Tensor x = torch::from_blob(dataCHW, {1, 3, 320, 320}, options);
    t_out = module.forward({x}).toTensor();

    delete[] dataCHW;

    return true;
}

bool inferSalience(unsigned char *dataAlbedo, int albedoWidth, int albedoHeight,
                   torch::jit::IValue &t_out) {
    cv::Mat srcAlbedo(albedoHeight, albedoWidth, CV_8UC4, dataAlbedo);
    cv::Mat srcAlbedoRGB, albedoResized;

    albedoWidth = SALIENCE_DIM;
    albedoHeight = SALIENCE_DIM;

    cv::cvtColor(srcAlbedo, srcAlbedoRGB, cv::COLOR_RGBA2RGB);
    cv::resize(srcAlbedoRGB, albedoResized, cv::Size(albedoWidth, albedoHeight), 0, 0, cv::INTER_AREA);

    float *dataCHW = new float[3*SALIENCE_DIM*SALIENCE_DIM];
    for (std::size_t i = 0; i < albedoHeight; i++) {
        for (std::size_t j = 0; j < albedoWidth; j++) {
            dataCHW[0 * albedoHeight * albedoWidth + albedoWidth*i + j] =
                    (float) (albedoResized.data[3*(i * albedoWidth + j) + 0]) / 255.0f;
            dataCHW[0 * albedoHeight * albedoWidth + albedoWidth*i + j] =
                    (dataCHW[0 * albedoHeight * albedoWidth + albedoWidth*i + j] - 0.485f) / 0.229f;

            dataCHW[1 * albedoHeight * albedoWidth + albedoWidth*i + j] =
                    (float) (albedoResized.data[3*(i * albedoWidth + j) + 1]) / 255.0f;
            dataCHW[1 * albedoHeight * albedoWidth + albedoWidth*i + j] =
                    (dataCHW[1 * albedoHeight * albedoWidth + albedoWidth*i + j] - 0.456f) / 0.224f;

            dataCHW[2 * albedoHeight * albedoWidth + albedoWidth*i + j] =
                    (float) (albedoResized.data[3*(i * albedoWidth + j) + 2]) / 255.0f;
            dataCHW[2 * albedoHeight * albedoWidth + albedoWidth*i + j] =
                    (dataCHW[2 * albedoHeight * albedoWidth + albedoWidth*i + j] - 0.406f) / 0.225f;
        }
    }

    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
    torch::Tensor x = torch::from_blob(dataCHW, {1, 3, SALIENCE_DIM, SALIENCE_DIM}, options);
    t_out = module2.forward({x});

    delete[] dataCHW;

    return true;
}

void fillHoles(cv::Mat &in, cv::Mat &out) {
    cv::Mat thresh;
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;

    cv::threshold(in, thresh, 20, 255, cv::THRESH_BINARY);
    cv::findContours(thresh, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_NONE);

    out = in.clone();

    cv::fillPoly(out, contours, cv::Scalar(255,255,255));
}

bool initTexture() {
//    stbi_set_flip_vertically_on_load(true);

    string albedoFullFilename = resourceDirectory + albedoFilename;
    string depthFullFilename = resourceDirectory + depthFilename;

    int albedoWidth, albedoHeight, albedoChn;
    unsigned char *dataAlbedo = stbi_load(albedoFullFilename.c_str(), &albedoWidth, &albedoHeight, &albedoChn, STBI_rgb_alpha);

    float ratio = (float) albedoWidth / albedoHeight;
    scale = glm::vec3(45.0f*ratio, 45.0f, 8.0f);
    scale2 = glm::vec3(40.0f*ratio, 40.0f, 4.0f);
    pointOfRotation = glm::vec3(scale.x / 2.0f, scale.y * 2.0f * heightNorm, scale.z);

    if (dataAlbedo == NULL) {
        LOGE("%s not found", albedoFullFilename.c_str());
        return false;
    }


    GL_CHECK(glGenTextures(1, &albedoTextureID));
    GL_CHECK(glBindTexture(GL_TEXTURE_2D, albedoTextureID));

    GL_CHECK(glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA,
                          albedoWidth, albedoHeight, 0,
                          GL_RGBA, GL_UNSIGNED_BYTE, dataAlbedo));

    /* Set texture mode. */
    GL_CHECK(glGenerateMipmap(GL_TEXTURE_2D));
    GL_CHECK(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR));
    GL_CHECK(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)); /* Default anyway. */
    GL_CHECK(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE));
    GL_CHECK(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE));

    int width, height, chn;
    unsigned char *data = NULL;

    float *salienceData = NULL;

    torch::jit::IValue t_out2;
    bool rs = inferSalience(dataAlbedo, albedoWidth, albedoHeight, t_out2);
    const torch::Tensor &tensor = t_out2.toTuple()->elements()[0].toTensor();
    if (rs == true) {
        int salienceHeight = tensor.sizes()[2];
        int salienceWidth = tensor.sizes()[3];

        salienceData = (float *) tensor.data_ptr();
    }

    torch::Tensor t_out;
    rs = inferDepth(dataAlbedo, albedoWidth, albedoHeight, salienceData, t_out);
    if (rs == true) {
        height = t_out.sizes()[1];
        width = t_out.sizes()[2];
        chn = 1;

        float *depthData = (float *) t_out.data_ptr();
        std::size_t totalDim = height * width;
        float min = depthData[0];
        float max = depthData[0];

        data = new unsigned char[height * width];

        for (std::size_t i = 1; i < totalDim; i++) {
            if (min > depthData[i]) {
                min = depthData[i];
            }
            if (max < depthData[i]) {
                max = depthData[i];
            }
        }

        for (std::size_t i = 0; i < totalDim; i++) {
            depthData[i] = 255 * (depthData[i] - min) / (max - min);
            data[i] = (unsigned char) depthData[i];
        }
    }

    cv::Mat thresh, threshDilated, threshEroded;
    // CREATE THRESH
    unsigned char salienceDataNorm[SALIENCE_DIM*SALIENCE_DIM];
    std::size_t lastI = 0;
    for (std::size_t i = 0; i < SALIENCE_DIM*SALIENCE_DIM; i++) {
        if (salienceData != NULL) {
            salienceDataNorm[i] = (unsigned char) (salienceData[i] * 255);
        } else {
            salienceDataNorm[i] = 255;
        }

        if (salienceData[i] > 0.5) {
            lastI = i;
        }
    }

    heightNorm = (float) lastI / SALIENCE_DIM;

    cv::Mat salience(SALIENCE_DIM, SALIENCE_DIM, CV_8UC1, salienceDataNorm);
    cv::Mat salienceResized, salienceBlur, salienceFilled;
    cv::resize(salience, salienceResized, cv::Size(width, height), 0, 0, cv::INTER_LINEAR);

    fillHoles(salienceResized, salienceResized);
    cv::dilate(salienceResized, salienceResized,
               cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7)));
    cv::GaussianBlur(salienceResized, salienceResized, cv::Size(5, 5), 0);
    cv::erode(salienceResized, thresh,
              cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(11, 11)));
    // END CREATE THRESH

    unsigned char *dataCopy = new unsigned char[width * height];
    unsigned char *dataCopy2 = new unsigned char[width * height];
    memcpy(dataCopy, data, width*height*sizeof(unsigned char));
    memcpy(dataCopy2, data, width*height*sizeof(unsigned char));

    qsort(dataCopy, (std::size_t) width*height, sizeof(unsigned char), compare);
    unsigned char median = (dataCopy[width*height / 2]);
    unsigned char minimum = dataCopy[0];
    unsigned char maximum = dataCopy[width*height - 1];
    delete[] dataCopy;

    int mean = 0;
    std::size_t count = 0;

    for (std::size_t i = 0; i < width*height; i++) {
        if (thresh.data[i] > 200) {
            mean += data[i];
            count++;
        }
    }

    mean = (unsigned char) (0.5f*mean / count);

    float C = 0.01f;
    float C2 = 0.01f;
    for (std::size_t i = 0; i < width*height; i++) {
        float dataF = (float) data[i];
        float dataLog;// = dataF * (log2(C*dataF + 1.0f) / log2(C * maximum + 1.0f));

        if (data[i] >= mean || thresh.data[i] > 200) {
            dataLog = dataF * (log2(C * dataF + 1.0f) / log2(C * maximum + 1.0f));
        } else {
            dataLog = dataF * pow(10.0f, C2*dataF+1.0f)/pow(10.0f, C2*maximum+1.0f);
        }

        if (dataLog < 0)
            dataLog = 0;
        if (dataLog > 255)
            dataLog = 255;

        data[i] = (unsigned char) dataLog;
    }


    cv::Mat src(height, width, CV_8UC1, data);
    cv::Mat dst;

    // sharpen image
    cv::bilateralFilter(src, dst, 15, 150, 150);

    // for contour detection
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    //cv::threshold(dst, thresh, median, 255, cv::THRESH_BINARY);

    cv::dilate(thresh, threshDilated,
                        cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(9, 9)));
    cv::dilate(thresh, threshEroded,
               cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3)));
    cv::Mat threshFiltered;
    cv::bilateralFilter(threshEroded, threshFiltered, 15, 75, 75);

//    threshEroded = thresh.clone();
//    cv::GaussianBlur(threshEroded, threshEroded, cv::Size(9, 9), 0);


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
    GL_CHECK(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE));
    GL_CHECK(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE));

    //stbi_image_free(data);
    delete[] data;

    GL_CHECK(glGenTextures(1, &maskTextureID));
    GL_CHECK(glBindTexture(GL_TEXTURE_2D, maskTextureID));

    GL_CHECK(glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE, width, height, 0, GL_LUMINANCE, GL_UNSIGNED_BYTE, threshFiltered.data));

    /* Set texture mode. */
    GL_CHECK(glGenerateMipmap(GL_TEXTURE_2D));
    GL_CHECK(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR));
    GL_CHECK(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)); /* Default anyway. */
    GL_CHECK(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE));
    GL_CHECK(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE));

    glGenBuffers(NUM_VBOS, vboBG);
    glBindBuffer(GL_ARRAY_BUFFER,
                 vboBG[VERTEX]);
    glBufferData(GL_ARRAY_BUFFER,
                 sizeof(Vertex) * _bgVertices.size(),
                 &_bgVertices[0],
                 GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vboBG[INDEX]);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER,
                 sizeof(unsigned int) * _bgIndices.size(),
                 &_bgIndices[0],
                 GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

    cv::Mat albedoInpainted, threshResized, albedoResized;
    cv::Mat srcAlbedo(albedoHeight, albedoWidth, CV_8UC4, dataAlbedo);
    cv::Mat srcAlbedoRGB, depthInpainted;

    cv::cvtColor(srcAlbedo, srcAlbedoRGB, cv::COLOR_RGBA2RGB);

    cv::resize(srcAlbedoRGB, albedoResized, cv::Size(width, height), 0, 0, cv::INTER_LINEAR);
    //cv::resize(thresh, threshResized, cv::Size(albedoWidth, albedoHeight), 0, 0, cv::INTER_LINEAR);

    cv::inpaint(albedoResized, threshDilated, albedoInpainted, 11, cv::INPAINT_TELEA);
    cv::inpaint(dst, threshDilated, depthInpainted, 11, cv::INPAINT_TELEA);
    cv::Mat src2(height, width, CV_8UC1, dataCopy2);

    cv::Mat bigmask, smallmask;
    cv::compare(src2, mean, bigmask, cv::CMP_GE);
    cv::bitwise_not(bigmask, smallmask);

    cv::Mat albedo1, albedo2, albedoRes;
    albedoInpainted.copyTo(albedo1);
    cv::GaussianBlur(albedo1, albedo1, cv::Size(11, 11), 0);
    albedoRes = albedoInpainted.clone();
    for (std::size_t i = 0; i < width*height; i++) {
        if (smallmask.data[i] == 255) {
            albedoRes.data[3*i] = albedo1.data[3*i];
            albedoRes.data[3*i+1] = albedo1.data[3*i+1];
            albedoRes.data[3*i+2] = albedo1.data[3*i+2];
        }
    }

    GL_CHECK(glGenTextures(1, &albedoBGTexture));
    GL_CHECK(glBindTexture(GL_TEXTURE_2D, albedoBGTexture));

    GL_CHECK(glTexImage2D(GL_TEXTURE_2D, 0, internalFormat[albedoChn],
                          albedoRes.size().width, albedoRes.size().height, 0,
                          pixelFormat[albedoChn], GL_UNSIGNED_BYTE, albedoRes.data));
    delete[] dataCopy2;

    /* Set texture mode. */
    GL_CHECK(glGenerateMipmap(GL_TEXTURE_2D));
    GL_CHECK(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR));
    GL_CHECK(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)); /* Default anyway. */
    GL_CHECK(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE));
    GL_CHECK(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE));

    GL_CHECK(glGenTextures(1, &depthTextureBGID));
    GL_CHECK(glBindTexture(GL_TEXTURE_2D, depthTextureBGID));

    GL_CHECK(glTexImage2D(GL_TEXTURE_2D, 0, internalFormat[chn], width, height, 0, pixelFormat[chn], GL_UNSIGNED_BYTE, depthInpainted.data));

    /* Set texture mode. */
    GL_CHECK(glGenerateMipmap(GL_TEXTURE_2D));
    GL_CHECK(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR));
    GL_CHECK(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)); /* Default anyway. */
    GL_CHECK(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE));
    GL_CHECK(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE));

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

        AndroidPlatform::getAndroidAsset(env, resourceDirectory.c_str(), initNet.c_str());
        AndroidPlatform::getAndroidAsset(env, resourceDirectory.c_str(), predictNet.c_str());

        AndroidPlatform::getAndroidAsset(env, resourceDirectory.c_str(), initNet2.c_str());
        AndroidPlatform::getAndroidAsset(env, resourceDirectory.c_str(), predictNet2.c_str());

        AndroidPlatform::getAndroidAsset(env, resourceDirectory.c_str(), modelPath.c_str());
        AndroidPlatform::getAndroidAsset(env, resourceDirectory.c_str(), u2netp_modelPath.c_str());

        // init networks
        //initCaffe2();
        initTorch();

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
