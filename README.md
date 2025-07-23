# Візуальна навігація для дрона
Ідея зробити щось схоже як на відео з 6-ої хв.  https://youtu.be/Ib0Qtl6ELHQ?si=6VesvbHNXm2n5zfw&t=382 

Весь проєкт поділений на етапи. Кожний етап має свою гілку в git.
Також в кожному етапі є завдання, які треба виконати.
Подивитися рішення можна в гілці наступного етапу.

## Як корисуватися
1. Клонуйте репозиторій
```bash
git clone https://github.com/robotics-students-ua/basic_drone_vio.git
```
2. Перейдіть до директорії проекту
```bash
cd basic_drone_vio
```
3. змінуть гілку на перший етап
```bash
git checkout 1_planar_quad
```
4. Запустіть симуляцію
```bash
python planar_quad/dynamics.py
```
за потреби встановіть requirements
```bash
pip install -r requirements.txt
```

## Як доповнити
1. Зробіть форк репозиторію
2. Створіть нову гілку для вашого доповнення
```bash
git checkout -b my_feature_branch
```
3. Внесіть зміни до коду
4. Додайте зміни до індексу
```bash
git add .
```
5. Зробіть коміт з описом змін
```bash
git commit -m "Додано нову функціональність"
```
6. Запуште зміни до вашого форку
```bash
git push origin my_feature_branch
```
7. Створіть pull request до основного репозиторію



## Етап 1
1. Симуляція дрона в 2D. Ручний політ. Додавайння візуальних рис.

файл: `planar_quad/dynamics.py` описує динаміку квадрокоптера в 2D, де рух відбувається в площині XZ (NED - North-East-Down). 

файл: `planar_quad/dynamics_online.py` - це онлайн-симуляція, яка використовує OpenCV для візуалізації динаміки квадрокоптера.
Додано можливість ручного керування дрона за допомогою стрілок на клавіатурі. Керування тягою та моментами напряму. Щоб було ясно навіщо теорія керування. 

файл: `planar_quad/dynamics_online_vision.py` - додано камеру на дрон, яка дивиться вниз на бінарний паттерни на землі. Треба ще доробити правильно переспективну проєкцію та візуалізацію. Це чудовий перший крок для розуміня візуальної навігації.

### Динаміка квадрокоптера в 2D

$$
\begin{align}
m \ddot{x} &= -(u_1 + u_2)\sin\theta \\
m \ddot{z} &= mg -(u_1 + u_2)\cos\theta \\
I \ddot{\theta} &= r (u_1 - u_2)
\end{align}
$$
where:
- \( m \) is the mass of the quadrotor
- \( I \) is the moment of inertia
- \( g \) is the acceleration due to gravity
- \( r \) is the distance from the center of mass to the point where the forces are applied
- \( u_1 \) and \( u_2 \) are the thrust forces applied by the quadrotor's motors
- \( \theta \) is the angle of the quadrotor with respect to the vertical axis
- \( \ddot{x} \) is the acceleration in the x-direction
- \( \ddot{z} \) is the acceleration in the z-direction

here we use NED (North-East-Down) coordinate system, where:
- \( x \) is the North direction
- \( z \) is the Down direction
- \( y \) is the East direction (toward us in planar quadrotor dynamics)