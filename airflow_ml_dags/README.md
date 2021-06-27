# airflow_ml_dags
**Installation and Usage:**
~~~
#Настройка переменых, созданных в UI Airflow
export FERNET_KEY=$(python -c "from cryptography.fernet import Fernet; 
FERNET_KEY = Fernet.generate_key().decode(); print(FERNET_KEY)")
~~~

~~~
#Запуск со сборкой контейнеров
docker-compose up --build
~~~
**Tests**
~~~
docker-compose up -d --build
docker ps
CONTAINER ID        IMAGE                COMMAND                  CREATED             STATUS              PORTS                    NAMES
b2c9213f694e        airflow-docker       "/usr/bin/dumb-init …"   13 seconds ago      Up 11 seconds       8080/tcp                 airflow_ml_dags_scheduler_1
docker exec -it b2c9213f694e bash
pip install pytest
pytest tests
~~~
**Project Structure:**

    ├── README.md          <- The top-level README for developers using this project.
    │
    ├── dags               <- contains all of airflow`s DAGs.
    │
    ├── images             <- contains docker images
    │
    ├── tests              <- Code for testings DAGs
    │
    └── docker-compose.yaml <- contains configuration
**Самооценка:**
1.  Поднят контейнер airflow локально, используя docker compose. 
    Реализован dag, который генерирует данные для обучения модели - 5 баллов
2. Реализован dag, который обучает модель еженедельно, используя данные за текущий день. - 10 баллов
    - подготовка данных для обучения - нормализация
    - распил их на train/val
    - обучение модели на train (сохранение в /data/models/{{ ds }} 
    - валидация модели на val (сохранение метрики к модельке)
3.  Реализован dag, который использует модель ежедневно  - 5 баллов
    - принимает на вход данные из пункта 1 (data.csv)
    - считывает путь до модельки из airflow variables(идея в том, что когда нам нравится другая модель и мы хотим ее на прод 
    - делает предсказание и записывает их в /data/predictions/{{ds }}/predictions.csv
4. Реализованы сенсоры на то, что данные готовы для дагов тренировки и обучения  - 3 балла
5. Все даги реализованы только с помощью DockerOperator  - 10 баллов
6. Тестирование  дагов  - 5 баллов
7. Самооценка - 1 балл

Итого: 39 баллов
