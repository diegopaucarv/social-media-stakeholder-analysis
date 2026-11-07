# Análisis de framing – agua privada, AdAM y gobernanza (Alto Minho / Viana do Castelo)

Datos: `facebook-posts-scraper` (30 filas, 26 posts con texto) y `facebook-comments-scraper` (694 comentarios, 29 posts distintos). Codificación manual de los 694 comentarios (lectura completa) y cálculo estadístico con Python (scipy 1.17).

> **Nota:** tu mensaje empieza con "Mi output final ha de ser…" y se corta ahí. Construí la entrega a partir de la descripción de la tarea (CSV de comentarios seleccionados + posts clasificados + línea de tiempo + estadística + faltantes). Si el formato final que tenías en mente es otro, dímelo y lo reformateo desde estos mismos archivos.

---

## 1. Contenido de la entrega

| Archivo                                                                                         | Qué es                                                                                                                                                                                                                                                                                                                                                                  |
| ----------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `01_comentarios_seleccionados.csv`                                                              | **185 comentarios** que sustentan un frame/hilo: marco, marco secundario, hilo, postura (defiende / contra / matiza), nivel de evidencia (N = nuclear, A = adyacente), nota de argumento, texto, fase de la línea de tiempo, relevancia, id/nombre/género/URL del usuario, página o grupo, post al que responde y tema/marco del post, enlace permanente al comentario. |
| `02_comentarios_todos_codificados.csv`                                                          | Los 694 comentarios con variables mínimas y `marco_codigo` (SIN = no encaja en ningún frame). Base de la estadística y de auditoría.                                                                                                                                                                                                                                    |
| `03_posts_clasificados.csv`                                                                     | 29 posts con comentarios (+ filas de posts sin comentarios): tipo de fuente, tema, **marco al que se relaciona el post**, marco dominante en sus comentarios, reacciones/compartidos.                                                                                                                                                                                   |
| `04_usuarios_perfil.csv`                                                                        | Todos los autores: actividad, likes/respuestas recibidos, marcos/hilos que defienden, localidades mencionadas, nivel ecosistémico y rol **solo si el propio texto lo evidencia**, `candidato_contacto`.                                                                                                                                                                 |
| `05_fases_linea_de_tiempo.csv`                                                                  | Fases por post (ver §3).                                                                                                                                                                                                                                                                                                                                                |
| `06_cola_revision_humana.csv`                                                                   | Cola de revisión humana: comentarios marcados por el clasificador (confianza baja o motivo de duda).                                                                                                                                                                                                                                                                    |
| `07_estadistica_chi2_cramer.csv`, `08_tablas_contingencia.txt`, `08b_residuos_ajustados_T1.csv` | Resultados estadísticos, tablas y residuos.                                                                                                                                                                                                                                                                                                                             |
| `framing/`, `scripts/`                                                                          | Pipeline reproducible (`python -m framing`) y scripts del estudio (`preparar_lotes_gemini.py`, `unir_clasificacion_gemini.py`). La codificación manual está en `01_` y `02_`.                                                                                                                                                                                           |

**Columnas eliminadas:** ~1.500 columnas técnicas del scraper (attachments/_, video/dash/shaka, feedback/_, top_reactions/_, collaborators/_, sharedPost/*, fotos de perfil, tracking, `comments/0..5/*` anidados —duplicaban filas ya presentes—, etc.). Se conservaron solo las que sirven al objetivo.

**Ética:** son datos públicos de personas privadas. Incluí `user_id` y nombre porque los pediste, pero también un seudónimo (`U###`); para cualquier informe compartido o publicado usa el seudónimo y no reproduzcas nombres.

---

## 2. Cómo se correlacionan las dos bases

Clave natural: `comentarios.facebookId` = ID numérico del post = `posts.postId`. Al leerlos como texto (no numérico, para no perder dígitos) coincide directo en el **75,6 %** de los comentarios. El resto falla porque las URL están truncadas y varían (`pfbid…`); lo resolví con `inputUrl` + `postTitle` (que en la base de comentarios trae el texto del post) y lo verifiqué leyendo cada texto. Resultado:

- 29 posts con comentarios ↔ 29 filas del archivo de posts (los 15 posts de AdAM, 3 del gabinete municipal, 2 reels, 1 foto, 1 medio, 3 perfiles, 1 político, 1 municipio y 3 grupos).
- 1 fila de posts (`search/top/?q=montaria coliformes alto minho`) devolvió `no_items`: es una URL de búsqueda, no un post; no aporta datos.
- **Duplicado:** el post de Manuel J. Gomes existe en su perfil (70 comentarios) y en el grupo _Praia D'Amorosa_ (8): mismo texto, misma fecha. Trátalo como un solo discurso al contar "posts".

---

## 3. Metodología aplicada (y ampliada)

**Unidad de análisis:** comentario. **Excluidos de las pruebas:** 60 respuestas institucionales (AdAM / Câmara) y 49 comentarios sin texto (stickers/imágenes) → **585 comentarios ciudadanos con texto**.

**Libro de códigos.** Cada comentario seleccionado recibe: (a) _marco_ F1–F4, (b) _marco secundario_ si el argumento es mixto, (c) _hilo argumental_, (d) _postura_ (defiende el frame / lo contradice / matiza) y (e) _nivel_ (N: reproduce casi literalmente el argumento objetivo; A: evidencia adyacente del mismo frame). Añadí la postura "contra" porque los contra-argumentos son parte del debate (p. ej. quienes explican la espuma como fenómeno natural).

| Marco                                                | Hilos (n comentarios)                                                                                                                                                                                                                                                     |
| ---------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| F1 Falsa seguridad de agua propia                    | H1 tradición/experiencia (2) · H2 uso autónomo acotado (2) · H3 análisis periódico como condición (3) · H4 depende del lugar (1)                                                                                                                                          |
| F2 Resistencia tarifaria / anti-centralización       | H1 cargos fijos y consumo cero (33) · H2 gestión municipal previa y concelhos no adheridos (24) · H3 salida individual pozo/mina/desconexión (15) · H4 movilización y reversión (19) · H5 precio vs. servicio (22)                                                        |
| F3 Desconfianza institucional / hipocresía ambiental | H1 negación oficial vs. sospecha de esgoto (14) · H2 alerta de AdAM leída como marketing (4) · H3 cierre y abandono de fuentes públicas (2) · H4 contaminación industrial/extractiva (7) · **H5 hipocresía de las campañas de ahorro (15) – hilo emergente, no previsto** |
| F4 Riesgos invisibles / fallas de red                | H1 calidad organoléptica: cloro, turbia, tierra (9) · H2 presión/continuidad/roturas (11) · H3 captación en río contaminado, Lima (2)                                                                                                                                     |

**Relevancia.** `score = likes + 2 × respuestas`. _Alta_ ≥ 6 (76 comentarios), _media_ 3–5 (69), _baja_ < 3 (549).

**Línea de tiempo y fases.** Por post, ordenado por fecha: la fase 1 arranca en el primer comentario; **cada comentario ciudadano de relevancia alta abre una fase nueva** que dura hasta el siguiente relevante o el final. Salen 92 fases (media 3,2 por post). `05_fases…csv` da inicio/fin, comentario ancla, likes y marco dominante de cada fase.

**Criterio de selección:** se seleccionó todo comentario que reproduce el argumento objetivo (N) o lo sustenta de forma adyacente (A), priorizando los más relevantes cuando había muchos casi idénticos (p. ej. insultos genéricos a AdAM quedaron fuera, salvo que aportaran un hilo).

---

## 4. Hallazgos

**Cobertura de frames (comentarios ciudadanos, 185 seleccionados):** F2 = 113 · F3 = 42 · F4 = 22 · F1 = 8 como marco principal (+17 donde F1 aparece como marco secundario → 25 comentarios / 21 autores tocan F1).

**F1 – Falsa seguridad de la fuente propia: presente pero distinto a la hipótesis.**

- Todo el F1 primario está en un único post (el aviso de AdAM sobre captações particulares, 21 comentarios).
- El argumento "no necesita análisis porque es transparente, corre siempre y nadie enfermó" aparece **solo de forma parcial**: un usuario dice que bebió años de poço y fontes "nunca adoeci" (c. 284); otro defiende una mina de 300 años potable frente a la obligación de conectarse (c. 36); otro dice no tener miedo de que no sea pura (c. 275).
- Lo más frecuente es lo contrario: **piden o aceptan análisis** ("as pessoas podem analisar em laboratório", c. 277; "desde que analisada periodicamente", c. 292; c. 419 pide muestras para el caso de Âncora). Rechazan la _prohibición/obligación_, no el control.
- Los comentarios más apoyados del hilo (c. 278, 17 likes; c. 280, 11) **no defienden la calidad del pozo, atacan al emisor** (AdAM) → F1 se desplaza a F3. Es el hallazgo central sobre F1: en estos datos la confianza en el agua propia se sostiene más por **desconfianza institucional y autonomía** que por una creencia sobre la pureza.
- Nitratos, coliformes y bacterias **no aparecen ni una vez** en los comentarios ciudadanos (solo en el texto del post de AdAM).

**F2 – Resistencia tarifaria: el frame dominante.** El agravio central es estructural: cargos con consumo cero (Porto 7,80 € vs. Viana 18,37 € según el post fuente), saneamiento aplicado a agua que no va al colector, RSU en casas vacías y comparación con concelhos que no entraron (Monção, Melgaço, Ponte da Barca) o con la gestión municipal previa (SMAS). El argumento "mantener pozo propio para evitar cobros" aparece en 15 comentarios (F2_H3), pero con un matiz importante: **la salida no funciona**: pagan igual sin consumir, el reglamento exige conexión donde hay red, desconectar es difícil o cuesta 180 € reconectar (c. 423, 586, 587, 589, 593). No hay comentarios que hablen de _reactivar_ pozos; sí de _mantenerlos_ (c. 210, 573, 585, 588, 487).

**F3 – Desconfianza institucional: más amplia que la formulación.**

- Coincide con la hipótesis: el post del vereador que niega que la espuma de Âncora sea contaminación ("nada a ver con descargas de esgotos") recibe sospecha vecinal ("a natureza não faz esgoto", "o problema é sempre no mesmo sítio") y contra-explicaciones científicas (surfactantes, materia orgánica) que otros aceptan. Un comentario pide análisis documentados (c. 419).
- Fontes públicas: c. 25 (19 likes) atribuye el cierre de fontes a que no facturaban; c. 280 (11 likes) recuerda que antes las autarquias las analizaban y colocaban carteles de "imprópria". **No hay en los datos ningún aviso municipal sobre fuentes públicas**, así que la reacción al aviso no se puede medir todavía.
- Hilo emergente **F3_H5** (15 comentarios): la campaña "cierra la torneira" se lee como hipocresía porque AdAM deja correr fugas durante días (c. 51, 36 likes, el más apoyado de toda la base).

**F4 – Riesgos invisibles:** predominan la calidad _visible_ (cloro/lejía, agua turbia, "como leche", tierra en la bañera) y las fallas de presión/continuidad; los "invisibles" (nitratos, bacterias) no aparecen. El vínculo explícito con buscar una alternativa subterránea es raro: c. 45 ("até para cozinhar tenho de ir à fonte"), c. 284. También aparece que la red se capta en el Lima "com problemas de poluição" (c. 577).

**Línea de tiempo.** En el post de captações particulares (F1) el debate arranca con posturas sobre analizar el agua y, a partir del primer comentario de alta relevancia (c. 278), pasa a la crítica al emisor y a la fiabilidad de la red; el post de Âncora se abre con la explicación oficial y se convierte (fase 2, 16 comentarios) en un debate mayoritariamente F3. En los posts de perfiles ciudadanos de 2026 (consumo cero) el discurso es F2 en todas las fases y se llena de comparaciones territoriales y de reversión política.

---

## 5. Estadística (chi² con corrección de Yates y V de Cramér)

Universo: 585 comentarios ciudadanos con texto. **Yates solo existe para tablas 2×2**; para tablas RxC uso chi² de Pearson + V de Cramér (más V corregida por sesgo, IC bootstrap 95 % y % de celdas con esperado < 5) y, para cada frame, una descomposición uno-contra-resto en 2×2 con Yates, Fisher exacto y ajuste de Holm.

| Prueba                                                                            | χ² (gl)         | p                     | V de Cramér          | Lectura                                                                                                                                                                                                                                      |
| --------------------------------------------------------------------------------- | --------------- | --------------------- | -------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| T1 Marco × tipo de espacio (AdAM / grupo / municipal-político / perfil ciudadano) | 105,3 (12)      | < 0,001               | 0,245 (IC 0,20–0,30) | Efecto moderado. Residuos: F2 **infra**representado en AdAM (−6,8) y **sobre**representado en perfiles ciudadanos (+7,3) y grupos (+3,2); F3 sobre en municipal/político (+4,0), ausente en perfiles (−3,3); F4 se concentra en AdAM (+3,0). |
| T1b F2 vs resto × AdAM vs otros                                                   | 44,4 (1, Yates) | < 0,001; Holm < 0,001 | φ = 0,28             | Robusto.                                                                                                                                                                                                                                     |
| T2 Marco × periodo del post                                                       | 44,3 (4)        | < 0,001               | 0,275                | F2: 35 % de los comentarios en 2026 vs. 14 % antes (Yates 32,0; p < 0,001; Holm < 0,001). **Confundido con T1** (todos los posts de 2026 son perfiles/grupos/municipales).                                                                   |
| T3 Marco × raíz/respuesta                                                         | 2,9 (4)         | 0,57                  | 0,07                 | Sin asociación.                                                                                                                                                                                                                              |
| T4 Marco × relevancia alta (solo codificados)                                     | 1,1 (3)         | 0,79                  | 0,08                 | Ningún frame gana más tracción que otro.                                                                                                                                                                                                     |
| T5 Marco × género (M/F)                                                           | 18,6 (4)        | < 0,001               | 0,18                 | Solo F2 (M 24 % vs. F 13 %; Yates 11,2; Holm p = 0,014).                                                                                                                                                                                     |
| T6 Hilos de F2 × espacio (exploratorio)                                           | 113,5 (12)      | < 0,001               | 0,58                 | H1 (consumo cero) casi solo en perfiles (28/33); H5 (precio vs. servicio) en AdAM (20/22); H4 (movilización) en grupos y municipal. 55 % de celdas < 5: solo descriptivo.                                                                    |
| T7 Marco × postura                                                                | 21,2 (6)        | 0,002                 | 0,24                 | Los contra-argumentos se concentran en F3 (9) y F2 (8); F4 ninguno.                                                                                                                                                                          |
| T8 Sensibilidad (1 comentario por autor y marco)                                  | 50,3 (9)        | < 0,001               | 0,33                 | Mismas conclusiones que T1.                                                                                                                                                                                                                  |

**Advertencias que hay que declarar:**

1. **Independencia:** los comentarios se agrupan en 29 posts y algunos autores repiten (T8 lo mitiga). Las diferencias por "espacio" o "periodo" son diferencias entre muy pocos posts (p. ej. 3 perfiles en 2026), no entre poblaciones. Con más posts convendría un modelo multinivel (logit mixto).
2. **Frames raros:** F1 tiene 8 casos; sus pruebas tienen ≥ 50 % de celdas con esperado < 5 (usa Fisher; en T1b F1 vs resto p_Fisher = 0,025, pero Holm no lo sostiene: 0,81).
3. **Sesgo de codificación:** los comentarios "SIN marco" incluyen insultos genéricos a AdAM y quejas de atención al cliente que podrían considerarse F2 _débil_; F2 está subestimado en AdAM y mi criterio es conservador. Por eso T4 se recalculó solo con codificados (la selección manual tuvo en cuenta likes; usar SIN habría sido circular).
4. **Un solo codificador (yo):** sin fiabilidad intercodificador. Antes de publicar, una segunda persona debería recodificar una muestra (≈ 20–25 % de los 185 + una muestra de SIN) y calcular κ de Cohen.
5. **Dependencia del sesgo del scraper:** las URL se eligieron a mano; los resultados describen _estos_ posts, no "Viana" en general.

---

## 6. Qué nos falta según la metodología del estudio

**Tarea 1 – Grupos y comunidades**

- **Solo 2 de los 6 grupos** que enviaste aparecen en los datos (`1010586782296426` Praia D'Amorosa y `700848426998968` SOS Serra D'Arga e Alto Minho). Faltan `244913175530233`, `535527626983000`, `254801216109855`, `7479391655408844`.
- Del listado de páginas **solo aparecen** `adp.aguasdoaltominho`, `gabinete.deimprensa` y `radiogeicefm`. Sin datos: Empresa Águas do Alto Minho (no…), Viana TV, Olhar Viana, Canal Viana, Ambiente Viana, Antena Minho, Meteo Trás-os-Montes, SSTMVC, S. Ambiental, Viana ao Minuto, Rádio Vale do Minho, Notícias de Caminha, COREMA Património, Associação Rio Neiva, SOS Serra d'Arga (página), ZERO ASTS e los perfiles por ID. Aparecen otros que **no estaban en tu lista** (Município de Valença, juntospelocastro, perfiles personales), probablemente por la búsqueda.
- **Asambleas/Juntas de freguesia y grupos cívicos de vecinos** (lo que pide la tarea): no hay ninguno. No se identifica la freguesia de forma sistemática (solo 60 comentarios con localidad mencionada; heurística en `localidad_mencionada`, hay que revisarla).
- **Relevancia local de la monitorización de fuentes privadas:** los comentarios permiten inferir dependencia de pozos (p. ej. "nas freguesias uma grande maioria só tem água do poço", c. 377), pero **faltan datos externos**: cobertura de la red por freguesia, hogares con captación privada, alertas sanitarias históricas (USP Alto Minho, ULS, APA/ERSAR). Sin eso el índice de prioridad por freguesia no se puede calcular.

**Tarea 2 – Extracción de contactos y evidencia**

- Hay enlaces permanentes por comentario (`url_comentario`) pero **no capturas de pantalla**; las URL de Facebook pueden cambiar o borrarse.
- **Water Stewards, profesores de ciencias, dirigentes de asociaciones de regantes/compartes, presidentes de Junta:** el texto solo permite marcar usuarios con pozo/mina/nascente propios (microsistema), autores de posts/organizadores (mesosistema) y actores institucionales; no aparece ningún profesor ni guardián formal. `candidato_contacto` es una lista **a validar**, no una lista de invitados.
- No hay datos de USP Alto Minho ni del Laboratório de Saúde Ambiental de la ULS (exo/macrosistema): solo se ven sus efectos indirectos.

**Brechas del corpus para los 4 frames**

- **F1:** con 8 casos principales el frame no es analizable estadísticamente. Falta material de comunidades rurales con captación propia (grupos de freguesias, grupos de regantes, foros de emigrantes). Sugiero búsquedas por poço, mina, furo, nascente, análise de água, coliformes/nitratos en Facebook y grupos locales; la búsqueda enviada (`montaria coliformes alto minho`) volvió vacía.
- **F3:** faltan los propios avisos municipales/USP sobre fuentes públicas ("imprópria para consumo") y las reacciones a ellos; y comunicados de ETAR/descargas (SMSBVC, AdAM, APA).
- **F4:** ningún comentario menciona nitratos o bacterias; hay que comprobar si esta ausencia es real (discurso técnico ausente en redes) o del recorte.
- Cobertura temporal desigual: 376 comentarios de 2021, 123 de 2022, 161 de 2026 y muy pocos entre 2023 y 2025.

**Metodología pendiente**

1. Fiabilidad intercodificador (κ) y refinamiento del libro de códigos (sobre todo F1: separar "seguridad experiencial" de "autonomía/anti-obligación", que hoy se solapan con F2/F3).
2. Análisis de red de respuestas (quién responde a quién) y de repetición de autores; unidad de análisis adicional: autor.
3. Modelo multinivel (comentario dentro de post) cuando haya más posts; hoy solo 29.
4. Recoger el contexto de exposición (comentarios totales vs. capturados: el scraper devuelve una parte; `commentsCount` solo está en 140 filas) y las reacciones/compartidos de cada post para ponderar.
5. Tratamiento ético: evaluación ética/RGPD, seudonimización, protocolo de contacto (los "candidatos" son usuarios que no consintieron ser contactados).
6. Triangulación con entrevistas y observación en campo (el estudio es participativo; el discurso en Facebook no equivale a la práctica de uso del agua).
