# Sum26 Manipulator

Симулятор манипулятора в MuJoCo для курса по робототехнике. В проекте есть модель UR10e с гриппером, Gymnasium-совместимое окружение, вспомогательный класс кинематики на Pinocchio и пример управления в пространстве суставов.

## Требования

- Рекомендуется Linux.
- Python 3.10.
- Рабочий OpenGL для рендера MuJoCo.
- Conda или Python `venv`.

Зависимости описаны в двух файлах:

- `environment.yml` для установки через conda.
- `requirements.txt` для установки через pip.

## Установка Через Conda

Из корня репозитория:

```bash
conda env create -f environment.yml
conda activate manipulator
```

## Установка Через Venv

Из корня репозитория:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Быстрая Проверка

Проверить синтаксис основных файлов:

```bash
python -m py_compile simulator/env.py joint_space_example.py
```

Проверить, что MuJoCo и Pinocchio импортируются:

```bash
python -c "import mujoco, pinocchio; print('MuJoCo and Pinocchio OK')"
```

## Запуск Симулятора

Запустить пример управления в пространстве суставов:

```bash
python joint_space_example.py
```

Пример создаёт окружение из `simulator/env.py`, загружает сцену `simulator/robot/scene.xml`, делает `reset()` и запускает цикл управления. Рендер задаётся параметром `render_mode` в примере:

```python
render_mode="all"       # окно viewer + изображения с камер
render_mode="human"     # только окно viewer
render_mode="rgb_array" # только изображения с камер
render_mode=None        # без рендера
```

## Режимы Управления

Окружение поддерживает два режима управления рукой:

```python
control_mode="joint_position"
```

В этом режиме первые 6 значений action задают целевые углы суставов в радианах.

```python
control_mode="joint_velocity"
```

В этом режиме первые 6 значений action задают целевые скорости суставов в `rad/s`. Окружение в памяти перенастраивает актуаторы руки в скоростные servo-регуляторы и может добавлять компенсацию гравитации. Последнее значение action управляет гриппером в обоих режимах.

## Кинематика

`PinKinematics` в `simulator/env.py` предоставляет:

- `solve_fk(joints)` для прямой кинематики: позиция end-effector и углы Эйлера.
- `solve_ik(position, euler, joints)` для обратной кинематики.
- `solve_jacobian(joints)` для матрицы Якоби end-effector.

Пример:

```python
from simulator.env import PinKinematics, get_robot_xml_path

kinematics = PinKinematics(
    model_path=get_robot_xml_path("ur10e2f85.xml"),
    ee_name="gripper_base",
)
```

## Возможные Проблемы

Если окно viewer не открывается, сначала попробуй запустить без рендера:

```python
render_mode=None
```

На headless-машинах MuJoCo может требовать настройку EGL или OSMesa. На локальном Linux-компьютере с установленными OpenGL-драйверами стандартный GLFW viewer обычно работает без дополнительных настроек.
