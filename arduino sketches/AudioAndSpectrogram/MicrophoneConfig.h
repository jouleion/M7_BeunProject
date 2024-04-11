// MicrophoneConfig.h
#ifndef MICROPHONECONFIG_H
#define MICROPHONECONFIG_H

#define I2S_MIC_CHANNEL I2S_CHANNEL_FMT_ONLY_LEFT
#define pin_i2s_sd 5
#define pin_i2s_ws 6
#define pin_i2s_sck 7

extern i2s_config_t i2s_config;
extern i2s_pin_config_t i2s_mic_pins;

i2s_config_t i2s_config = {
    .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
    .sample_rate = SAMPLE_RATE,
    .bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT,
    .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
    .communication_format = I2S_COMM_FORMAT_I2S,
    .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
    .dma_buf_count = 4,
    .dma_buf_len = 1024,
    .use_apll = false,
    .tx_desc_auto_clear = false,
    .fixed_mclk = 0
};

i2s_pin_config_t i2s_mic_pins = {
    .bck_io_num = pin_i2s_sck,
    .ws_io_num = pin_i2s_ws,
    .data_out_num = I2S_PIN_NO_CHANGE,
    .data_in_num = pin_i2s_sd
};

#endif // MICROPHONECONFIG_H
