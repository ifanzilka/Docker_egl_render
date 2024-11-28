# Docker_egl_render

### Сборка образа

        docker build -t egl_render .

### Запуск образа
        docker run -it --gpus all egl_render egl_render_container

### Вход в образ
        docker exec -it egl_render_container bash

### Запсук скрипта для инициализации виртуального дисплея

        bash entrypoint.sh

### Запсук рендеринга

        vglrun python3 blend_multi_universal.py --path_with_glb=./tmp_glb/ --path_save=./tmp_render


