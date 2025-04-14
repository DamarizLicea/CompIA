V1 — Semana del 31 de marzo al 4 de abril
Accuracy promedio de 0.5.

El modelo fue capaz de detectar al menos una clase correctamente.

V2 — Semana del 7 al 11 de abril
Implementé una versión personalizada de ColorJitter inspirada en (Ref2), con el objetivo de hacer el data_augmentation más robusto.
Sin embargo, el accuracy cayó de 0.5 a un máximo de 0.02, con fluctuaciones muy abruptas y comportamiento inestable.

Se reorganizó la distribución del dataset para reflejar mejores prácticas:

80% entrenamiento

10% validación

10% prueba

Distribución actual del dataset:

Total de etiquetas: 75 clases

Imágenes de entrenamiento: 5199

Imágenes de validación: 650

Imágenes de prueba: 650

Encontré un paper relevante (Ref1) sobre clasificación de mariposas con más de 100 clases usando Inception-V3.
Aún estoy investigando cómo implementar esa arquitectura y los tres tipos de data_augmentation que utilizaron.
Por ahora, el enfoque está en optimizar un solo tipo de aumento antes de añadir más complejidad.

Aumenté el número de épocas de entrenamiento de 10 a 100, lo cual representa un cambio importante en el ciclo de pruebas.

Gráficos de precisión
Antes de modificar el data_augmentation:

![image](https://github.com/user-attachments/assets/6af0b676-5c32-4dc0-b62a-5f6dc140547e)

Después de la modificación agresiva:

![image](https://github.com/user-attachments/assets/e1ca6bf7-e3be-4687-8d2c-aff2bf254c59)

Con ajustes menos agresivos:
![image](https://github.com/user-attachments/assets/20704f3d-f9d4-455a-b995-f69e89f9ac38)


Referencias
[Ref1] Paper sobre clasificación de mariposas con más de 100 clases:
https://dl.futuretechsci.org/id/eprint/73/1/9443-Article%20Text-30154-5-10-20240615.pdf

[Ref2] Estudio reciente sobre técnicas avanzadas de data augmentation:
https://doi.org/10.48550/arXiv.2502.18691
