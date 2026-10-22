# Основи дрона: модель, керування, оцінка стану

Навчальний проєкт про те, як літає квадрокоптер і як він «знає», де він є.
Ідемо від простого до складного: спочатку плоский (2D) квадрокоптер, потім повний 3D.


## Що тут вивчаємо

1. **Динаміка** — рівняння руху квадрокоптера (тяга, момент, нахил).
2. **Керування** — каскад регуляторів: acro → stab → velocity → position.
3. **Датчики** — гіроскоп і акселерометр, і чому IMU самої по собі замало.
4. **Камера** — модель pinhole і положення дрона з зображення.
5. **Фільтр Калмана** — від одного числа до EKF, що поєднує IMU і камеру.

## Структура

```
2d_quad/            плоский квадрокоптер (x, z, θ)
  chapters/         розділи 1–5 (.qmd джерело, .html для читання)
  1.dynamics.py     динаміка, графіки
  2.dynamics_online.py          керування з клавіатури
  3.dynamics_online_vision.py   + камера
  vision_1.py       положення з зображення
  ekf.py            EKF: IMU + камера
  simulator.py      інтерактивний симулятор
  web/              той самий симулятор у браузері
3d_quad/            повний 3D квадрокоптер
  quad.py, rigid_body.py        динаміка
  stab_controller.py, altitude_controller.py   регулятори
  batch.py, batch_altitude.py   симуляція з графіками
  sim_mujoco.py     симуляція з рендером у MuJoCo і камерою
```

## Як почати

```bash
git clone https://github.com/robotics-students-ua/basic_drone_vio.git
cd basic_drone_vio
pip install -r requirements.txt scipy
```

Читайте лекції по порядку (відкрийте `.html` у браузері) і запускайте відповідний код:

```bash
cd 2d_quad
python 1.dynamics.py         # лекція 1
python simulator.py          # лекції 2–3, керування: WASD / стрілки, r — скинути
python vision_1.py           # лекція 4
python ekf.py                # лекція 5
```

Симулятор у браузері (без встановлення Python-пакетів):

```bash
cd 2d_quad/web
python3 -m http.server 8000
# відкрийте http://localhost:8000/sim/
```

3D квадрокоптер:

```bash
pip install mujoco tqdm
cd 3d_quad
python batch.py              # стабілізація кута, графіки
python batch_altitude.py     # утримання висоти, графіки
python sim_mujoco.py         # інтерактивно: w/s a/d q/e y/h, t — утримання висоти
```

## Етапи в гілках

Старіші етапи збережені в окремих гілках (`1_planar_quad`, `2_control_planar_quad`,
`3_state_estimation`). У кожній гілці є свої завдання, а рішення — у гілці наступного етапу.

```bash
git checkout 1_planar_quad
```

## Системи координат

Скрізь використовуємо **NED**: вісь `z` дивиться вниз, тому дрон на висоті 10 м має `z = -10`.
