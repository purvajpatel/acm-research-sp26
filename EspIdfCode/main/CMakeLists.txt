idf_component_register(SRCS "main.c" "PrintFunctions.c"
                    PRIV_REQUIRES spi_flash
                    INCLUDE_DIRS "."
                    REQUIRES esp32-camera
                    )