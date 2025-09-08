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
3. змініть гілку на бажаний етап. В кожному етапі своє readme.md
```bash
git checkout 1_planar_quad
git checkout 2_control_planar_quad
```
4. Запустіть симуляцію
```bash
python planar_quad/1.dynamics.py
python planar_quad/2.dynamics_online.py
```
за потреби встановіть requirements
```bash
pip install -r requirements.txt
```




Для симуляції камери, також треба склонувати  
git clone https://github.com/hronoses/GameOfDronesDev

і запустити в Unity (тека unity2022) сцена drone_vio

TODO інтегрувати все в один репозиторій;%: