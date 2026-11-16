idf_component_register(SRCS "esp32_lcd_main.c"
                       REQUIRES esp_lcd esp_driver_uart esp_driver_spi esp_driver_gpio)
