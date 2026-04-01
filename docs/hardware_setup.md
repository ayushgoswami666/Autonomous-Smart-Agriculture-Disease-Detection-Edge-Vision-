# Hardware Setup Guide

## Required Components
- Raspberry Pi 4 (4GB RAM recommended)
- Raspberry Pi Camera Module v2
- MicroSD Card (32GB minimum)
- Power Supply (5V 3A)
- Optional: 7 inch touchscreen display

## Assembly Steps
1. Insert MicroSD card with Raspberry Pi OS installed
2. Connect Camera Module to CSI port
3. Connect power supply
4. Boot and enable camera in settings

## Camera Enable
```
sudo raspi-config
```
Go to Interface Options > Camera > Enable

## Test Camera
```
raspistill -o test.jpg
```
