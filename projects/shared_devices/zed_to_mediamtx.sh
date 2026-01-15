#!/bin/bash

# ZED Camera to MediaMTX streaming script
# This script sends left and right ZED camera streams to MediaMTX via RTSP
# Uses NVIDIA GPU encoding for low latency

MEDIAMTX_HOST="localhost"
MEDIAMTX_PORT="8554"
# Resolution options: 0=HD2K(2208x1242), 1=HD1080(1920x1080), 2=HD1200(1920x1200), 3=HD720(1280x720), 5=VGA(672x376)
# use "ZED_Explorer" app to test different resolutions and frame rates
RESOLUTION="1" 
FRAMERATE="30"   # 30fps = stable, 60fps = smoother (currently not working)
BITRATE="12000"  # Higher bitrate = better quality (kbps)
GOP_SIZE=${FRAMERATE}    # GOP = framerate for balance
QUEUE_SIZE="1"   # Minimize buffering for lowest latency (1-3 buffers)

# Kill any existing pipelines
pkill -f "gst-launch.*zedsrc"

echo "Starting ZED stereo camera pipeline with NVIDIA GPU encoding..."
echo "Resolution Option: ${RESOLUTION} @ ${FRAMERATE}fps | Bitrate: ${BITRATE}kbps"
echo "Streaming to:"
echo "  Left:  rtsp://${MEDIAMTX_HOST}:${MEDIAMTX_PORT}/zed/left"
echo "  Right: rtsp://${MEDIAMTX_HOST}:${MEDIAMTX_PORT}/zed/right"
echo ""

# Single pipeline that opens camera once and splits into left/right streams
# Optimized for low latency and high quality
/usr/bin/gst-launch-1.0 -e \
    zedsrc stream-type=2 camera-resolution=${RESOLUTION} camera-fps=${FRAMERATE} ! \
    video/x-raw,format=BGRA,framerate=${FRAMERATE}/1 ! \
    zeddemux name=demux \
    demux.src_left ! queue max-size-buffers=${QUEUE_SIZE} leaky=downstream ! videoconvert ! video/x-raw,format=I420 ! \
        nvh264enc preset=low-latency-hq rc-mode=cbr bitrate=${BITRATE} gop-size=${GOP_SIZE} \
        zerolatency=true bframes=0 spatial-aq=true aq-strength=8 qos=true ! \
        video/x-h264,profile=main,stream-format=byte-stream ! h264parse ! \
        queue max-size-buffers=${QUEUE_SIZE} leaky=downstream ! \
        rtspclientsink location=rtsp://${MEDIAMTX_HOST}:${MEDIAMTX_PORT}/zed/left protocols=tcp latency=0 \
    demux.src_aux ! queue max-size-buffers=${QUEUE_SIZE} leaky=downstream ! videoconvert ! video/x-raw,format=I420 ! \
        nvh264enc preset=low-latency-hq rc-mode=cbr bitrate=${BITRATE} gop-size=${GOP_SIZE} \
        zerolatency=true bframes=0 spatial-aq=true aq-strength=8 qos=true ! \
        video/x-h264,profile=main,stream-format=byte-stream ! h264parse ! \
        queue max-size-buffers=${QUEUE_SIZE} leaky=downstream ! \
        rtspclientsink location=rtsp://${MEDIAMTX_HOST}:${MEDIAMTX_PORT}/zed/right protocols=tcp latency=0

echo ""
echo "Pipeline stopped."
