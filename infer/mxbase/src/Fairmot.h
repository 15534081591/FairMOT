#include <memory>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

#include "MxBase/DeviceManager/DeviceManager.h"
#include "MxBase/DvppWrapper/DvppWrapper.h"
#include "MxBase/ModelInfer/ModelInferenceProcessor.h"
#include "MxBase/PostProcessBases/ObjectPostProcessBase.h"
#include "MxBase/Tensor/TensorContext/TensorContext.h"
#include "PostProcess/FairmotMindsporePost.h"

struct InitParam {
  uint32_t deviceId;
  std::string labelPath;
  uint32_t classNum;
  float iouThresh;
  float scoreThresh;

  bool checkTensor;
  std::string modelPath;
};

struct ImageShape {
  uint32_t width;
  uint32_t height;
};

class Fairmot {
 public:
  APP_ERROR Init(const InitParam &initParam);
  APP_ERROR DeInit();
  APP_ERROR Inference(const std::vector<MxBase::TensorBase> &inputs,
                      std::vector<MxBase::TensorBase> &outputs);
  APP_ERROR PostProcess(const std::vector<MxBase::TensorBase> &inputs,
                        MxBase::JDETracker &tracker, MxBase::Files &file);
  APP_ERROR Process(const std::string &imgPath);
  APP_ERROR ReadImageCV(const std::string &imgPath, cv::Mat &imageMat,
                        ImageShape &imgShape);
  APP_ERROR ResizeImage(const cv::Mat &srcImageMat, cv::Mat &dstImageMat,
                        ImageShape &imgShape);
  APP_ERROR CVMatToTensorBase(const cv::Mat &imageMat,
                              MxBase::TensorBase &tensorBase);
  APP_ERROR GetMetaMap(const ImageShape imgShape,
                       const ImageShape resizeimgShape,
                       MxBase::JDETracker &tracker);
  void WriteResult(const std::string &result_filename,
                   std::vector<MxBase::Results *> results);

 private:
  std::shared_ptr<MxBase::DvppWrapper> dvppWrapper_;
  std::shared_ptr<MxBase::ModelInferenceProcessor> model_;
  std::shared_ptr<MxBase::FairmotMindsporePost> post_;
  MxBase::ModelDesc modelDesc_;
  uint32_t deviceId_ = 0;
  double inferCostTimeMilliSec = 0.0;
};
