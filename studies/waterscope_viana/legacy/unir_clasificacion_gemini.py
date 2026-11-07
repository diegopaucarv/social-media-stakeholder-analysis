#!/usr/bin/env python3
"""
Une la clasificación de Gemini con el CSV ORIGINAL de comentarios (Apify) y reconstruye
las mismas variables del análisis previo: relevancia, fases, marco_etiqueta, hilo_etiqueta, etc.

Uso:
  python unir_clasificacion_gemini.py \
      --comentarios dataset_facebook-comments-scraper_XXXX.csv \
      --gemini "salida_gemini/*.json" \
      --posts 03_posts_clasificados.csv \
      --out salida/

--gemini acepta uno o varios patrones (.json array, .jsonl o .csv). La clave de unión es el
`id` ORIGINAL del comentario (columna `id` del scraper).
--posts (opcional) = 03_posts_clasificados.csv; aporta tema_post, marco_post, tipo_fuente,
espacio, autor_post y periodo por id_post. Sin él, se usan valores de respaldo (postTitle, etc.).
"""
import argparse, glob, json, os, re, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')

# ---------------- constantes (idénticas al análisis previo) ----------------
LAB = {'F1': 'F1 Falsa seguridad de agua propia',
       'F2': 'F2 Resistencia tarifaria/centralización',
       'F3': 'F3 Desconfianza institucional/hipocresía ambiental',
       'F4': 'F4 Riesgos invisibles/fallas de red',
       'SIN': 'Sin marco'}
HL = {'F1_H1': 'Tradición/experiencia como garantía',
      'F1_H2': 'Uso autónomo acotado (riego/huerta/no potable)',
      'F1_H3': 'Análisis periódico como condición de seguridad',
      'F1_H4': 'Ambivalencia: el riesgo depende del lugar',
      'F2_H1': 'Cargos fijos y consumo cero (desproporción)',
      'F2_H2': 'Gestión municipal previa / concelhos no adheridos (anti-centralización)',
      'F2_H3': 'Salida individual: pozo/mina/fonte/desconexión',
      'F2_H4': 'Movilización y reversión',
      'F2_H5': 'Precio vs. servicio (lecturas, estimativas, atención)',
      'F3_H1': 'Negación oficial vs. sospecha vecinal (espuma/esgoto)',
      'F3_H2': 'Alerta de AdAM leída como marketing/negocio',
      'F3_H3': 'Cierre/abandono de fuentes públicas y su control',
      'F3_H4': 'Contaminación industrial/extractiva y complicidad',
      'F3_H5': 'Hipocresía de campañas de ahorro (fugas/riego público)',
      'F4_H1': 'Calidad organoléptica de la red (cloro, turbia, tierra)',
      'F4_H2': 'Fallas de presión/continuidad/roturas',
      'F4_H3': 'Captación de red en río contaminado (Lima)'}
INST = {'Águas do Alto Minho', 'Câmara Municipal de Viana do Castelo'}
LOC = ['Ponte de Lima', 'Viana do Castelo', 'Viana', 'Caminha', 'Valença', 'Arcos de Valdevez',
       'Ponte da Barca', 'Cerveira', 'Monção', 'Melgaço', 'Freixo', 'Mujães', 'Refoios',
       'Rebordões', 'Neiva', 'Afife', 'Âncora', 'Arca', 'Venade', 'Senhora da Luz', 'Ganhei',
       'Soutelo', 'Azevedo', 'Gaia', 'Porto']
UMBRAL_ALTA, UMBRAL_MEDIA = 6, 3          # score = likes + 2*respuestas
MARCOS_VALIDOS = {'F1', 'F2', 'F3', 'F4', 'SIN'}


# ---------------- lectura de la salida de Gemini ----------------
def _parse_json_text(t):
    t = t.strip()
    t = re.sub(r'^```(?:json)?\s*|\s*```$', '', t, flags=re.I)
    try:
        d = json.loads(t)
    except json.JSONDecodeError:
        d = [json.loads(l) for l in t.splitlines() if l.strip().startswith('{')]
    return d if isinstance(d, list) else [d]


def load_gemini(patterns):
    files = sorted({f for p in patterns for f in glob.glob(p)})
    if not files:
        raise SystemExit('No se encontró ningún archivo de Gemini con --gemini')
    frames = []
    for f in files:
        if f.lower().endswith('.csv'):
            frames.append(pd.read_csv(f, dtype=str).fillna(''))
        else:
            txt = open(f, encoding='utf-8').read()
            frames.append(pd.DataFrame(_parse_json_text(txt)))
    g = pd.concat(frames, ignore_index=True)
    defaults = {'es_institucional': False, 'marco': 'SIN', 'marco_secundario': '', 'hilo': '',
                'hilo_emergente_propuesto': '', 'postura': '', 'nivel': '', 'extracto_literal': '',
                'justificacion': '', 'confianza': 3, 'requiere_revision_humana': False, 'motivo_revision': ''}
    for k, v in defaults.items():
        if k not in g.columns:
            g[k] = v
    g['id'] = g['id'].astype(str).str.strip()
    g = g.drop_duplicates('id', keep='last')
    for k in ['marco', 'marco_secundario', 'nivel']:
        g[k] = g[k].fillna('').astype(str).str.strip().str.upper()
    for k in ['hilo', 'postura', 'justificacion', 'hilo_emergente_propuesto', 'motivo_revision']:
        g[k] = g[k].fillna('').astype(str).str.strip()
    g['postura'] = g.postura.str.lower()
    g['confianza'] = pd.to_numeric(g.confianza, errors='coerce').fillna(3).astype(int)
    g['extracto_literal'] = g.extracto_literal.fillna('')
    return g


def loc_mencionada(t):
    f = [l for l in LOC if re.search(r'\b' + re.escape(l) + r'\b', t, flags=re.I)]
    return '; '.join(dict.fromkeys(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--comentarios', required=True)
    ap.add_argument('--gemini', required=True, nargs='+')
    ap.add_argument('--posts', default=None)
    ap.add_argument('--out', default='salida')
    ap.add_argument('--min-confianza', type=int, default=1,
                    help='confianza mínima (1-3) para considerar un comentario "seleccionado"')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    # ---------------- carga y unión por id de comentario ----------------
    c = pd.read_csv(a.comentarios, low_memory=False, dtype={'facebookId': str, 'id': str})
    c['id'] = c['id'].astype(str).str.strip()
    g = load_gemini(a.gemini)
    cols_g = ['id', 'es_institucional', 'marco', 'marco_secundario', 'hilo', 'hilo_emergente_propuesto',
              'postura', 'nivel', 'extracto_literal', 'justificacion', 'confianza',
              'requiere_revision_humana', 'motivo_revision']
    g = g[cols_g].rename(columns={'es_institucional': 'inst_gemini', 'justificacion': 'nota_argumento'})
    print(f'Comentarios originales: {len(c)} | filas Gemini: {len(g)} | '
          f'ids de Gemini encontrados en el original: {g.id.isin(c.id).mean():.1%}')
    c = c.merge(g, on='id', how='left')
    sin_clasif = c.marco.isna()
    print(f'Comentarios sin clasificación de Gemini (se tratan como SIN): {int(sin_clasif.sum())}')

    # ---------------- numeración de posts (igual que antes: URL ordenadas alfabéticamente) ----------------
    c = c.sort_values(['inputUrl', 'date'], kind='stable').reset_index(drop=True)
    c['cid'] = range(1, len(c) + 1)
    codes = {u: i + 1 for i, u in enumerate(c.inputUrl.unique())}
    c['pn'] = c.inputUrl.map(codes).astype(int)

    # ---------------- metadatos de posts ----------------
    if a.posts:
        pdf = pd.read_csv(a.posts, dtype=str)
        pdf = pdf[pd.to_numeric(pdf.id_post, errors='coerce').notna()].copy()
        pdf['pn'] = pdf.id_post.astype(int)
        for k in ['tipo_fuente', 'espacio', 'tema_post', 'marco_post', 'autor_post', 'periodo']:
            if k not in pdf.columns:
                pdf[k] = ''
        pdf = pdf[['pn', 'tipo_fuente', 'espacio', 'tema_post', 'marco_post', 'autor_post', 'periodo']]
    else:
        base = c.groupby('pn').agg(tema_post=('postTitle', 'first'), espacio=('groupTitle', 'first'),
                                   fmin=('date', 'min')).reset_index()
        base['tema_post'] = base.tema_post.fillna('').astype(str).str[:120]
        base['espacio'] = base.espacio.fillna('')
        base['tipo_fuente'] = ''
        base['marco_post'] = ''
        base['autor_post'] = ''
        base['periodo'] = np.where(base.fmin.astype(str).str[:4] == '2026', '2026', 'pre-2026')
        pdf = base.drop(columns='fmin')
        print('AVISO: sin --posts; tema_post/tipo_fuente/periodo son valores de respaldo (revísalos).')
    c = c.merge(pdf, on='pn', how='left')

    # ---------------- variables de comentario (idénticas al pipeline previo) ----------------
    c['fecha'] = pd.to_datetime(c.date)
    c['texto'] = c.text.fillna('')
    c['sin_texto'] = c.text.isna()
    c['likes'] = c.likesCount.fillna(0).astype(int)
    rep_obs = c.replyToCommentId.value_counts()
    c['n_respuestas_obs'] = c.id.map(rep_obs).fillna(0).astype(int)
    c['n_respuestas'] = np.maximum(c.n_respuestas_obs, c.commentsCount.fillna(0)).astype(int)
    c['score_relevancia'] = c.likes + 2 * c.n_respuestas
    c['relevancia'] = np.where(c.score_relevancia >= UMBRAL_ALTA, 'alta',
                               np.where(c.score_relevancia >= UMBRAL_MEDIA, 'media', 'baja'))
    c['autor'] = c['author/name']
    c['es_institucional'] = c.autor.isin(INST) | c.inst_gemini.fillna(False).astype(bool)
    c['es_autor_post'] = (c.autor == c.autor_post)
    c['tiene_adjunto'] = c['attachments/0/__typename'].notna() if 'attachments/0/__typename' in c else False
    c['profundidad'] = c.threadingDepth
    c['es_respuesta'] = (c.threadingDepth > 0).astype(int)
    c['genero'] = c['author/gender'].map({'MALE': 'M', 'FEMALE': 'F', 'NEUTER': 'no_declarado'}).fillna('desconocido') \
        if 'author/gender' in c else 'desconocido'
    c['perfil_url'] = c['author/url'] if 'author/url' in c else ''
    c['verificado'] = c['author/is_verified'] if 'author/is_verified' in c else ''
    pa = c['parentComment/author/name'] if 'parentComment/author/name' in c else pd.Series(np.nan, index=c.index)
    pr = c['parentReply/author/name'] if 'parentReply/author/name' in c else pd.Series(np.nan, index=c.index)
    c['comentario_padre_autor'] = pa.fillna(pr)

    # ---------------- línea de tiempo y fases ----------------
    c = c.sort_values(['pn', 'fecha', 'cid']).reset_index(drop=True)
    c['orden_cronologico'] = c.groupby('pn').cumcount() + 1
    fase = []
    for pn, gg in c.groupby('pn', sort=True):
        n, started = 0, False
        for _, r in gg.iterrows():
            if not started:
                n, started = 1, True
            elif r.relevancia == 'alta' and not r.es_institucional:
                n += 1
            fase.append((r.cid, n))
    c['fase'] = c.cid.map(dict(fase))

    # ---------------- marco / hilo / etiquetas ----------------
    c['marco'] = c.marco.fillna('SIN')
    c.loc[~c.marco.isin(MARCOS_VALIDOS), 'marco'] = 'SIN'          # valores inesperados -> SIN
    c.loc[c.es_institucional, 'marco'] = 'SIN'                      # respuestas institucionales
    for k in ['marco_secundario', 'hilo', 'hilo_emergente_propuesto', 'postura', 'nivel', 'nota_argumento',
              'extracto_literal', 'motivo_revision']:
        c[k] = c[k].fillna('')
    c['confianza'] = c.confianza.fillna(0).astype(int)
    c['requiere_revision_humana'] = c.requiere_revision_humana.fillna(False).astype(bool)
    # coherencia hilo-marco: si el hilo no pertenece al marco, se marca para revisión
    incoh = (c.marco != 'SIN') & (c.hilo != '') & (c.hilo != 'EMERGENTE') & (c.hilo.str[:2] != c.marco)
    c.loc[incoh, 'requiere_revision_humana'] = True
    c.loc[incoh, 'motivo_revision'] = (c.loc[incoh, 'motivo_revision'] + ' hilo no corresponde al marco').str.strip()
    c.loc[c.marco == 'SIN', ['hilo', 'postura', 'nivel', 'marco_secundario']] = ''
    c['seleccionado'] = (c.marco != 'SIN') & (c.confianza >= a.min_confianza)
    c['marco_etiqueta'] = c.marco.map(LAB)
    c['hilo_etiqueta'] = c.hilo.map(HL).fillna('')
    emerg = c.hilo == 'EMERGENTE'
    c.loc[emerg, 'hilo_etiqueta'] = 'EMERGENTE: ' + c.loc[emerg, 'hilo_emergente_propuesto']

    c['localidad_mencionada'] = c.texto.apply(loc_mencionada)
    ids = sorted(c['author/id'].astype(str).unique())
    pseud = {x: 'U' + str(i + 1).zfill(3) for i, x in enumerate(ids)}
    c['usuario_pseudonimo'] = c['author/id'].astype(str).map(pseud)

    # ---------------- salidas ----------------
    cols_sel = ['cid', 'pn', 'tipo_fuente', 'espacio', 'tema_post', 'marco_post', 'autor_post', 'fecha', 'fase',
                'orden_cronologico', 'relevancia', 'score_relevancia', 'likes', 'n_respuestas', 'profundidad',
                'es_respuesta', 'marco', 'marco_etiqueta', 'marco_secundario', 'hilo', 'hilo_etiqueta', 'postura',
                'nivel', 'nota_argumento', 'texto', 'author/id', 'usuario_pseudonimo', 'autor', 'genero',
                'verificado', 'perfil_url', 'es_autor_post', 'es_institucional', 'localidad_mencionada',
                'tiene_adjunto', 'commentUrl', 'inputUrl', 'comentario_padre_autor']
    extra = ['extracto_literal', 'confianza', 'requiere_revision_humana', 'motivo_revision', 'hilo_emergente_propuesto']
    rename = {'cid': 'id_comentario', 'pn': 'id_post', 'fecha': 'fecha_comentario', 'author/id': 'user_id',
              'autor': 'usuario_nombre', 'marco': 'marco_codigo', 'texto': 'texto_comentario',
              'commentUrl': 'url_comentario', 'inputUrl': 'url_post_key', 'profundidad': 'profundidad_hilo',
              'nivel': 'nivel_evidencia(N=nuclear,A=adyacente)', 'tipo_fuente': 'tipo_espacio',
              'espacio': 'pagina_o_grupo', 'autor_post': 'autor_del_post'}
    cols_sel = [k for k in cols_sel if k in c.columns]
    sel = c[c.seleccionado][cols_sel + extra].rename(columns=rename).sort_values(['id_post', 'fecha_comentario'])
    sel.to_csv(f'{a.out}/01_comentarios_seleccionados.csv', index=False, encoding='utf-8-sig')

    all_cols = ['cid', 'pn', 'tipo_fuente', 'espacio', 'fecha', 'fase', 'relevancia', 'score_relevancia', 'likes',
                'n_respuestas', 'profundidad', 'es_respuesta', 'marco', 'hilo', 'postura', 'nivel', 'author/id',
                'usuario_pseudonimo', 'autor', 'genero', 'es_institucional', 'es_autor_post', 'sin_texto', 'texto']
    c[all_cols + extra].rename(columns=rename).sort_values(['id_post', 'fecha_comentario']) \
        .to_csv(f'{a.out}/02_comentarios_todos_codificados.csv', index=False, encoding='utf-8-sig')

    ph = []
    for (pn, f), gg in c.groupby(['pn', 'fase']):
        gg = gg.sort_values('fecha')
        rel = gg[(gg.relevancia == 'alta') & (~gg.es_institucional)].head(1)
        cnt = gg[gg.seleccionado].marco.value_counts()
        ancla = (int(gg.iloc[0].cid) if f == 1 and gg.iloc[0].relevancia != 'alta'
                 else int(rel.cid.iloc[0]) if len(rel) else int(gg.iloc[0].cid))
        ph.append({'id_post': pn, 'tipo_espacio': gg.tipo_fuente.iloc[0], 'fase': f,
                   'inicio': gg.fecha.min().strftime('%Y-%m-%d %H:%M'), 'fin': gg.fecha.max().strftime('%Y-%m-%d %H:%M'),
                   'n_comentarios': len(gg), 'n_institucionales': int(gg.es_institucional.sum()),
                   'comentario_ancla_id': ancla, 'ancla_es_relevante': bool((f > 1) or gg.iloc[0].relevancia == 'alta'),
                   'likes_totales': int(gg.likes.sum()),
                   'marco_dominante': cnt.idxmax() if len(cnt) else 'SIN',
                   'n_F1': int(cnt.get('F1', 0)), 'n_F2': int(cnt.get('F2', 0)),
                   'n_F3': int(cnt.get('F3', 0)), 'n_F4': int(cnt.get('F4', 0))})
    pd.DataFrame(ph).to_csv(f'{a.out}/05_fases_linea_de_tiempo.csv', index=False, encoding='utf-8-sig')

    # cola de revisión humana
    rev = c[(c.requiere_revision_humana) | ((c.marco != 'SIN') & (c.confianza == 1))]
    rev[['cid', 'pn', 'marco', 'hilo', 'postura', 'nivel', 'confianza', 'motivo_revision', 'extracto_literal', 'texto']] \
        .rename(columns=rename).to_csv(f'{a.out}/06_cola_revision_humana.csv', index=False, encoding='utf-8-sig')

    print(f'Seleccionados: {int(c.seleccionado.sum())} | por marco: {c[c.seleccionado].marco.value_counts().to_dict()}')
    print(f'Relevancia: {c.relevancia.value_counts().to_dict()} | fases totales: {len(ph)} | a revisión humana: {len(rev)}')
    print(f'Archivos escritos en: {os.path.abspath(a.out)}')


if __name__ == '__main__':
    main()
