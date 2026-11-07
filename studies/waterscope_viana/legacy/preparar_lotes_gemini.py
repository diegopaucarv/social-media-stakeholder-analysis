#!/usr/bin/env python3
"""
PASO PREVIO a Gemini: calcula relevancia y fases con el CSV original y genera lotes listos
para el prompt (un post por lote, <= N comentarios, orden cronológico, con texto del comentario padre).

Uso:
  python preparar_lotes_gemini.py --comentarios dataset_facebook-comments-scraper_XXXX.csv \
      --posts 03_posts_clasificados.csv --out lotes/ --tam-lote 30

Salida:
  lotes/base_comentarios.csv      -> variables numéricas (relevancia, fase, orden) por comentario
  lotes/lote_P{post}_{k}.json     -> {"post_context": {...}, "comments_batch": [...]} para el bloque USER
"""
import argparse, json, os, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')

INST = {'Águas do Alto Minho', 'Câmara Municipal de Viana do Castelo'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--comentarios', required=True)
    ap.add_argument('--posts', default=None)
    ap.add_argument('--out', default='lotes')
    ap.add_argument('--tam-lote', type=int, default=30)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    c = pd.read_csv(a.comentarios, low_memory=False, dtype={'facebookId': str, 'id': str})
    c = c.sort_values(['inputUrl', 'date'], kind='stable').reset_index(drop=True)
    c['cid'] = range(1, len(c) + 1)
    codes = {u: i + 1 for i, u in enumerate(c.inputUrl.unique())}
    c['pn'] = c.inputUrl.map(codes).astype(int)

    # --- relevancia (igual que en el análisis) ---
    c['fecha'] = pd.to_datetime(c.date)
    c['likes'] = c.likesCount.fillna(0).astype(int)
    rep = c.replyToCommentId.value_counts()
    c['n_respuestas'] = np.maximum(c.id.map(rep).fillna(0), c.commentsCount.fillna(0)).astype(int)
    c['score_relevancia'] = c.likes + 2 * c.n_respuestas
    c['relevancia'] = np.where(c.score_relevancia >= 6, 'alta', np.where(c.score_relevancia >= 3, 'media', 'baja'))
    c['es_institucional'] = c['author/name'].isin(INST)

    # --- fases (los comentarios institucionales no abren fase) ---
    c = c.sort_values(['pn', 'fecha', 'cid']).reset_index(drop=True)
    c['orden_cronologico'] = c.groupby('pn').cumcount() + 1
    fase = {}
    for pn, g in c.groupby('pn'):
        n, st = 0, False
        for r in g.itertuples():
            if not st:
                n, st = 1, True
            elif r.relevancia == 'alta' and not r.es_institucional:
                n += 1
            fase[r.cid] = n
    c['fase'] = c.cid.map(fase)
    c[['cid', 'id', 'pn', 'fecha', 'orden_cronologico', 'fase', 'likes', 'n_respuestas', 'score_relevancia',
       'relevancia', 'es_institucional']].to_csv(f'{a.out}/base_comentarios.csv', index=False, encoding='utf-8-sig')

    # --- contexto del post ---
    ctx = {}
    if a.posts:
        p = pd.read_csv(a.posts, dtype=str)
        p = p[pd.to_numeric(p.id_post, errors='coerce').notna()]
        for r in p.itertuples():
            ctx[int(r.id_post)] = {'post_id': r.id_post, 'space_type': getattr(r, 'tipo_fuente', ''),
                                   'space_name': getattr(r, 'espacio', ''), 'post_date': getattr(r, 'fecha_post', ''),
                                   'post_text': getattr(r, 'post_texto_inicio', '')}
    # --- lotes: se excluyen institucionales y comentarios sin texto (no se clasifican) ---
    txt_by_id = c.set_index('id').text.to_dict()
    n_lotes = 0
    for pn, g in c.groupby('pn'):
        g = g[(~g.es_institucional) & g.text.notna()]
        for k in range(0, len(g), a.tam_lote):
            sub = g.iloc[k:k + a.tam_lote]
            batch = []
            for _, r in sub.iterrows():
                par = r.replyToCommentId
                batch.append({'id': r['id'], 'date': str(r['date']), 'threadingDepth': int(r.threadingDepth),
                              'likesCount': int(r.likes), 'commentsCount': int(r.n_respuestas),
                              'replyToCommentId': None if pd.isna(par) else par,
                              'parent_text': '' if pd.isna(par) else str(txt_by_id.get(par) or '')[:400],
                              'author/name': r['author/name'], 'text': r['text']})
            pc = ctx.get(pn, {'post_id': str(pn), 'space_type': '', 'space_name': '', 'post_date': '',
                              'post_text': str(g.postTitle.iloc[0]) if 'postTitle' in g else ''})
            json.dump({'post_context': pc, 'comments_batch': batch},
                      open(f'{a.out}/lote_P{pn:02d}_{k // a.tam_lote + 1}.json', 'w', encoding='utf-8'),
                      ensure_ascii=False, indent=1)
            n_lotes += 1
    print(f'Comentarios: {len(c)} | a clasificar: {int(((~c.es_institucional) & c.text.notna()).sum())} | lotes: {n_lotes}')
    print(f'Archivos en {os.path.abspath(a.out)}')


if __name__ == '__main__':
    main()
