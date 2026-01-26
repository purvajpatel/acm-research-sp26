idf_component_register(SRCS "main.cpp" "PrintFunctions.cpp" "model_data.cc"
                    PRIV_REQUIRES spi_flash
                    INCLUDE_DIRS "."
                    REQUIRES esp32-camera esp-tflite-micro
                    )