#include "Fairmot.h"
#include "MxBase/Log/Log.h"

namespace {
const uint32_t DEVICE_ID = 0;
}  // namespace

int main(int argc, char *argv[]) {
  int num = 2;
  if (argc <= num) {
    LogWarn << "Please input image path, such as './Fairmot_mindspore "
               "[om_file_path] [img_path]'.";
    return APP_ERR_OK;
  }

  InitParam initParam = {};
  initParam.deviceId = DEVICE_ID;

  initParam.checkTensor = true;

  initParam.modelPath = argv[1];
  auto inferFairmot = std::make_shared<Fairmot>();
  APP_ERROR ret = inferFairmot->Init(initParam);
  if (ret != APP_ERR_OK) {
    LogError << "Fairmot init failed, ret=" << ret << ".";
    return ret;
  }

  std::string imgPath = argv[2];
  ret = inferFairmot->Process(imgPath);
  if (ret != APP_ERR_OK) {
    LogError << "Fairmot process failed, ret=" << ret << ".";
    inferFairmot->DeInit();
    return ret;
  }
  inferFairmot->DeInit();
  return APP_ERR_OK;
}
