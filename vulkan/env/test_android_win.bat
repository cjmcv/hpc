adb shell "rm -r /data/local/tmp/pai/"

adb push ./bin/pai_vk /data/local/tmp/pai/bin/pai_vk
adb push ./shaders /data/local/tmp/pai/shaders

adb shell "chmod 777 -R /data/local/tmp/pai/ && cd /data/local/tmp/pai/ && ./bin/pai_vk"
