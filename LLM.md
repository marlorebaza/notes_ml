
## Procesamiento

Tokenización: Consiste en cortar el texto en unidades más pequeñas llamadas tokens (que pueden ser palabras, sílabas o partes de palabras) y asignar a cada una un número de identificación. Es el paso previo que permite traducir el lenguaje humano al formato numérico que el modelo procesa.

Embeddings: Es la conversión de cada token en un vector numérico multidimensional que captura su significado semántico y contexto. Permite al modelo entender conceptos abstractos y medir similitudes entre palabras (por ejemplo, reconocer que "perro" y "gato" están estrechamente relacionados).

TensorFlow Embedding Projector (projector.tensorflow.org) es una herramienta web interactiva desarrollada por Google para visualizar e inspeccionar datos de alta dimensión, especialmente los embeddings
https://projector.tensorflow.org/

## Consultas

### 1. ¿Cómo entender una quinta dimensión desde la perspectiva humana?

Una cuarta dimensión espacial se suele representar como un **teseracto** (un cubo desplegado o movido a lo largo de un cuarto eje perpendicular a los tres habituales).

Para entender la **quinta dimensión** (y de ahí en adelante hasta cientos de ellas), la física y las matemáticas nos enseñan a **dejar de intentar visualizar espacio físico** y empezar a pensar en **grados de libertad o variables independientes**:

* **Un punto (0D):** Solo posición fija.
* **1D (Línea):** Moverte adelante/atrás (Longitud).
* **2D (Plano):** Izquierda/derecha y adelante/atrás (Longitud + Ancho).
* **3D (Espacio):** Añades arriba/abajo (Longitud + Ancho + Altura).
* **4D (Espacio-Tiempo):** Añades el tiempo (cuándo ocurre el objeto en ese punto 3D).
* **5D (Espacio de Posibilidades / Coordenadas de Estado):** Imagina la 5D como la capacidad de moverte entre **líneas temporales alternativas** o **diferentes configuraciones de un sistema**.

Si estás en un punto 4D (tú, en tu habitación, a las 3:00 PM), la 5ª dimensión sería el eje que conecta *"tú a las 3:00 PM habiendo tomado café"* con *"tú a las 3:00 PM habiendo tomado té"*. No es una dirección hacia arriba o al lado, sino **una dirección de variación de un estado**.

En matemáticas y Machine Learning, una dimensión no es más que **una pregunta independiente que puedes hacer sobre un objeto**. Si un objeto necesita 500 respuestas numéricas para describirse por completo, vive en un espacio de 500 dimensiones.

---

### 2. ¿Cada dimensión se podría entender como una característica?

**Sí, exactamente.** Esa es la mejor intuición teórica: **1 Dimensión = 1 Característica o Variable Independiente.**

Si quisieras ubicar a todas las personas del mundo en un "espacio multidimensional", podrías asignar:

* Dimensión 1: Edad
* Dimensión 2: Estatura
* Dimensión 3: Ingresos
* Dimensión 4: Nivel educativo
* Dimensión 5: Gusto por la música clásica (de 0 a 1)

Con esas 5 características, cada persona se convierte en un **punto (vector) en un espacio de 5 dimensiones**. Las personas con gustos, edades e ingresos parecidos quedarán muy cerca unas de otras en ese espacio.

En los LLMs funciona igual, con la diferencia de que las características no las define un humano, sino que la red neuronal las descubre sola durante el entrenamiento.

---

### 3. ¿Cada palabra tiene solamente un vector?

Depende del tipo de modelo:

#### En modelos antiguos (Word2Vec, GloVe): SÍ, solo un vector por palabra

Tenían una tabla fija donde la palabra "banco" tenía **un solo vector**. Esto generaba un problema grave con la polisemia (palabras con varios significados):

* El vector de "banco" era un promedio extraño entre un *banco financiero* y un *banco para sentarse en el parque*.

#### En los LLMs modernos (Transformers como GPT-4, Llama, Claude): NO, el vector cambia según el contexto

Imagina la red neuronal como una autopista de varias capas por la que viajan los vectores:

```
Texto original  ──>  Vector Estático (Base)  ──>  Capas de Atención (Contexto)  ──>  Vector Dinámico Final
"banco a sacar dinero"   [0.1, -0.5, 0.8, ...]       Ajuste matemático por entorno      [0.9, 0.4, -0.2, ...] (Finanzas)

```

##### Step 1: El vector base estático

El modelo tiene una única tabla de búsqueda (*Look-up Table*). Al leer la palabra "banco", extrae siempre el mismo vector inicial. Este vector base no sabe en qué frase está; solo contiene un "resumen promedio" de todos los usos posibles de la palabra.

##### Step 2: La magia del mecanismo de atención (*Self-Attention*)

A medida que el vector avanza por las capas del Transformer, **interactúa matemáticamente con los vectores de las palabras vecinas** (como "dinero", "sacar" o "parque").

Para hacer esto, el modelo transforma cada vector en tres variantes mediante multiplicaciones de matrices:

1. **Query (Consulta):** ¿Qué tipo de información estoy buscando? *(Ej. "banco" busca pistas sobre su significado).*
2. **Key (Clave):** ¿Qué tipo de información ofrezco? *(Ej. "dinero" ofrece la pista de que la frase habla de finanzas).*
3. **Value (Valor):** El contenido de la información que se va a transferir.

##### Step 3: Ponderación y mezcla vectorial

El modelo calcula una **puntuación de afinidad** (mediante el producto escalar / similitud entre la *Query* de "banco" y las *Keys* de las demás palabras):

* La palabra **"dinero"** obtiene una puntuación de afinidad alta con **"banco"**.
* La palabra **"el"** o **"a"** obtiene una puntuación casi nula.

Luego, el vector de "banco" **absorbe una fracción de la información (los *Values*)** de las palabras con mayor afinidad.

##### Step 4: El vector resultante

Matemáticamente, lo que ocurre es una **suma ponderada de vectores**. Al sumar una porción del vector "dinero" al vector inicial de "banco", las coordenadas del vector "banco" **se desplazan espacialmente** en el espacio multidimensional.

Al salir de la última capa, el vector de "banco" ya no es el vector genérico de la tabla inicial: se ha transformado en un **vector contextualizado** ubicado en la región de *entidades financieras*.


Gemini usado: La idea es ir preguntandole cualquier duda y pedirle un ejemplo para montarlo en codelabs
https://gemini.google.com/app/402973e69b016dd2
Continua implementando el ejemplos que te generó en su ultima respuesta en colab
https://colab.research.google.com/drive/1n0M72D_N4t892UHCwHKGV74-KUnBK37A#scrollTo=ywAUZQSpCmiM

Continuar con el curso de LLMs de plazi 

Sigue viendo el video SOBRE OBsidian para llevar tus apuntes
https://www.youtube.com/watch?v=bt0JKhHcvn4

Revisa tbn este video sobre SDD para complementar lo aprendido en platzo
https://www.youtube.com/watch?v=5HaOxAAA5qI

https://colab.research.google.com/


los 4 campos obligatorios: SO, Metadata, By pass, volumetria
que la metadata corresponda a la descripcion
que la AAP este dado de alta


