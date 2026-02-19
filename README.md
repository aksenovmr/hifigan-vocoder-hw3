## Аннотация

Этот репозиторий представляет собой проект по реализации вокодера для генерации аудио из mel-спектрограмм.
В проекте реализован вокодер на основе статьи HiFi-GAN: ["Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis"](https://arxiv.org/pdf/2010.05646).
Обучение модели проводилось на датасете RUSLAN, состоящем из 22200 аудиозаписей на русском языке.
Основное применение обученной модели заключается в resynthesis, то есть в синтезировании аудио из mel-спектрограммы исходного аудио.
Датасет скачивается отдельно по [ссылке](https://ruslan-corpus.github.io/)

Подробнее о модели и проведенной работе, а также об анализе результатов можно ознакомиться в [отчете](https://github.com/aksenovmr/hifigan-vocoder-hw3/blob/main/report/%D0%9E%D0%A2%D0%A7%D0%95%D0%A2.ipynb)


## Запуск вокодера

Для проверки работоспособности вокодера можно воспользоваться следующей инструкцией.

### Для локальной проверки:

1. Клонирование репозитория
'''
git clone https://github.com/aksenovmr/hifigan-vocoder-hw3.git

cd hifigan-vocoder-hw3
'''
2. Активация виртуального окружения
'''
python -m venv .venv

source .venv/bin/activate # Linux/ Mac

.venv\Scripts\activate  # Windows
'''
3. Установка зависимостей
'''
pip install --upgrade pip

pip install -r requirements.txt
'''
4. Скачивание весов

Скачайте файл:

checkpoint-epoch120.pth из https://huggingface.co/aksenovmr/hifigan-vocoder-hw3/tree/main

Создайте папку для весов:
'''
mkdir -p checkpoints
'''
И поместите файл туда:
'''
mv checkpoint-epoch120.pth checkpoints/
'''
4. Запуск синтеза аудио
'''
python synthesize.py \
  --config src/configs/hifigan.yaml \
  --checkpoint checkpoints/checkpoint-epoch120.pth \
  --input_dir demo/mos_ground_truth \
  --output_dir demo/mos_samples
'''
Режим работы:

Audio -> Mel -> Vocoder -> Audio

### Для проверки в Google Colab:

1. Клонирование репозитория
'''
!git clone https://github.com/aksenovmr/hifigan-vocoder-hw3.git

%cd hifigan-vocoder-hw3
'''
2. Установка зависимостей
'''
!pip install --upgrade pip

!pip install -r requirements.txt

!pip install huggingface_hub soundfile
'''
3. Скачивание весов
'''
from huggingface_hub import hf_hub_download

CHECKPOINT_PATH = hf_hub_download(
    repo_id="aksenovmr/hifigan-vocoder-hw3",
    filename="checkpoint-epoch120.pth")

print("Checkpoint downloaded:", CHECKPOINT_PATH)
'''
4. Запуск синтеза аудио
'''  
!python synthesize.py \
  --config src/configs/hifigan.yaml \
  --checkpoint $CHECKPOINT_PATH \
  --input_dir demo/mos_ground_truth \
  --output_dir demo/mos_samples
'''

### Обучение модели

Для начала нужно: 

1) Создать папки data/RUSLAN в корне репозитория
2) Добавить датасет RUSLAN в распакованном виде по пути: data/RUSLAN/

Для запуска обучения:
'''
python train.py -cn hifigan writer=wandb trainer.n_epochs=120
'''
Для возобновления обучения:
'''
python train.py \
  -cn hifigan \
  writer=wandb \
  trainer.resume_from=checkpoint-epoch120.pth \
  trainer.n_epochs=150
'''
## Логи обучения

Обучение логируется в Weights & Biases. Отчет с графиками и аудио доступен по ссылке:

https://api.wandb.ai/links/aksenovmr-hse-university/ivxg45t3

Присутствуют следующие графики: train_loss, val_loss, grad_norm, steps_per_sec_train, steps_per_sec_val, epoch_train, а также MOS-аудио по эпохам.

Логи обучения в виде файла .log лежат в репозитории по пути [report/output.log](https://github.com/aksenovmr/hifigan-vocoder-hw3/blob/main/report/output.log)

## Демо
В проекте также присутствует jupyter-ноутбук [demo.ipynb](https://github.com/aksenovmr/hifigan-vocoder-hw3/blob/main/demo/%D0%94%D0%B5%D0%BC%D0%BE.ipynb) с демонстрацией работы синтезатора.

А именно:

1. Клонирование репозитория

2. Загрузка модели и весов

3. Демонстрация resynthesis

4. Воспроизведение MOS-примеров с демонстрацией качества

Ноутбук запускается в Google Colab.

Демо всей TTS системы проводилось в рамках написания отчета и находится в отдельном [файле](https://github.com/aksenovmr/hifigan-vocoder-hw3/blob/main/report/%D0%94%D0%B5%D0%BC%D0%BE_%D0%B8_%D0%B0%D0%BD%D0%B0%D0%BB%D0%B8%D0%B7_TTS.ipynb) с демонстрацией работы предобученной акустической модели, переводящей текст в аудио, а также вокодера, осуществляющего синтез речи на основе mel-спектрограмм, полученных из этих аудио.
