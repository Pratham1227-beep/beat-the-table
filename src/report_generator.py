"""
report_generator.py
───────────────────
Generates a styled PDF match report for a Beat the Table prediction.
Returns raw bytes that can be served via st.download_button().

Dependencies: fpdf2, matplotlib (already in requirements)
"""

import io
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches


# ── Color constants (RGB tuples) ───────────────────────────────────────────────
C_BG          = (8,  14, 28)
C_SURFACE     = (15, 23, 42)
C_ELEVATED    = (30, 41, 59)
C_BLUE        = (96, 165, 250)
C_RED         = (244, 63, 94)
C_AMBER       = (245, 158, 11)
C_TEXT        = (226, 232, 240)
C_MUTED       = (100, 116, 139)
C_WHITE       = (255, 255, 255)
C_GREEN       = (34, 197, 94)

POS_COLORS = {
    'GK':  C_AMBER,
    'CB':  C_BLUE, 'LB':  C_BLUE, 'RB':  C_BLUE,
    'LWB': C_BLUE, 'RWB': C_BLUE,
    'CDM': (16, 185, 129), 'CM':  (16, 185, 129),
    'CAM': (16, 185, 129), 'LM':  (16, 185, 129), 'RM':  (16, 185, 129),
    'LW':  C_RED,  'RW':  C_RED,  'ST':  C_RED,
}

SEV_COLORS = {
    '🔴 Critical': (220, 38,  38),
    '🟠 Warning':  (234, 88,  12),
    '🟡 Caution':  (202, 138,  4),
    '🟢 Solid':    (34,  197, 94),
}


def _fig_to_png_bytes(fig) -> bytes:
    """Save a matplotlib figure to a PNG byte buffer."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight',
                facecolor=fig.get_facecolor(), dpi=150)
    buf.seek(0)
    plt.close(fig)
    return buf.read()


def _pitch_figure(players_with_xy, dot_color_rgb=(96, 165, 250), highlights=None):
    """Render a portrait pitch matplotlib figure and return PNG bytes."""
    fig, ax = plt.subplots(figsize=(4, 5.6))
    fig.patch.set_facecolor('#0a1628')
    ax.set_facecolor('#0d7a3e')
    W, H = 100, 100
    lw = 1.5

    # Pitch markings
    ax.add_patch(patches.Rectangle((0, 0), W, H, linewidth=2, edgecolor='white', facecolor='none'))
    ax.plot([0, W], [H / 2, H / 2], color='white', linewidth=lw, alpha=0.7)
    ax.add_patch(patches.Circle((W / 2, H / 2), 9.5, edgecolor='white', facecolor='none', linewidth=lw, alpha=0.7))
    pa_w, pa_h = 56, 16
    ax.add_patch(patches.Rectangle(((W - pa_w) / 2, H - pa_h), pa_w, pa_h, linewidth=lw, edgecolor='white', facecolor='none', alpha=0.7))
    ax.add_patch(patches.Rectangle(((W - pa_w) / 2, 0), pa_w, pa_h, linewidth=lw, edgecolor='white', facecolor='none', alpha=0.7))

    dot_color = '#{:02x}{:02x}{:02x}'.format(*dot_color_rgb)

    for p in players_with_xy:
        px, py = p['x'], p['y']
        color = dot_color
        if highlights and p['name'] in highlights:
            color = highlights[p['name']]
            ax.plot(px, py, 'o', color=color, markersize=18, alpha=0.25)
        ax.plot(px, py, 'o', color=color, markersize=12, markeredgecolor='white', markeredgewidth=1.2, zorder=5)
        clean = p['name'].split(' ')[-1] if ' ' in p['name'] else p['name']
        ax.text(px, py - 6.5, f"{clean}\n{p['role']}", color='white', fontsize=5.5,
                ha='center', va='top', fontweight='bold', zorder=6,
                bbox=dict(facecolor='#00000099', edgecolor='none', boxstyle='round,pad=0.2'))

    ax.set_xlim(-3, 103)
    ax.set_ylim(-10, 108)
    ax.axis('off')
    plt.tight_layout(pad=0.2)
    return _fig_to_png_bytes(fig)


def generate_match_report(
    team_a: str,
    team_b: str,
    formation_a: str,
    formation_b: str,
    probs: dict,
    explanation: dict,
    starting_xi_a: list,
    starting_xi_b: list,
    mapped_a: list,
    mapped_b_mirror: list,
    weak_links: list,
    sim_res: dict,
    opp_style: str,
) -> bytes:
    """
    Generate a full PDF match report.

    Parameters
    ----------
    All parameters mirror the data already computed by app.py after the Analyse click.

    Returns
    -------
    bytes  — raw PDF content, ready for st.download_button()
    """
    try:
        from fpdf import FPDF
    except ImportError:
        raise ImportError("fpdf2 is required. Run: pip install fpdf2")

    # Pre-render pitch images
    pitch_a_bytes = _pitch_figure(mapped_a,   dot_color_rgb=C_BLUE)
    pitch_b_bytes = _pitch_figure(mapped_b_mirror, dot_color_rgb=(244, 63, 94))

    # ── Build PDF ──────────────────────────────────────────────────────────────
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    # ── Helper closures ────────────────────────────────────────────────────────
    def set_bg(r, g, b):
        pdf.set_fill_color(r, g, b)

    def text_color(r, g, b):
        pdf.set_text_color(r, g, b)

    def draw_rect(x, y, w, h, r, g, b):
        pdf.set_fill_color(r, g, b)
        pdf.rect(x, y, w, h, 'F')

    def section_header(title, icon=''):
        pdf.ln(6)
        draw_rect(10, pdf.get_y(), 190, 8, *C_ELEVATED)
        pdf.set_xy(12, pdf.get_y() + 1.5)
        pdf.set_font('Helvetica', 'B', 9)
        text_color(*C_BLUE)
        pdf.cell(0, 5, f'{icon}  {title}'.strip(), ln=True)
        pdf.ln(3)
        text_color(*C_TEXT)

    def small_tag(x, y, label, r, g, b):
        draw_rect(x, y, len(label) * 2.1 + 4, 5, r, g, b, )
        pdf.set_xy(x + 2, y + 0.7)
        pdf.set_font('Helvetica', 'B', 6.5)
        text_color(*C_WHITE)
        pdf.cell(0, 3.5, label)
        text_color(*C_TEXT)

    def prob_bar(label, pct, r, g, b, y_off=0):
        y = pdf.get_y() + y_off
        pdf.set_xy(12, y)
        pdf.set_font('Helvetica', '', 8)
        text_color(*C_MUTED)
        pdf.cell(40, 5, label)
        # Track
        draw_rect(52, y + 1.5, 100, 2, *C_ELEVATED)
        # Fill
        fill_w = max(0, min(100, pct * 100))
        draw_rect(52, y + 1.5, fill_w, 2, r, g, b)
        # Pct
        pdf.set_xy(155, y)
        pdf.set_font('Helvetica', 'B', 9)
        text_color(r, g, b)
        pdf.cell(0, 5, f'{pct:.1%}')
        text_color(*C_TEXT)
        pdf.ln(6)

    # ─────────────────────────────────────────────────────────────────────────
    # PAGE 1: COVER
    # ─────────────────────────────────────────────────────────────────────────
    pdf.add_page()
    set_bg(*C_BG)
    pdf.rect(0, 0, 210, 297, 'F')

    # Gradient-ish top banner
    draw_rect(0, 0, 210, 65, *C_SURFACE)

    # Rainbow top strip
    colors = [(59, 130, 246), (96, 165, 250), (56, 189, 248), (129, 140, 248)]
    for i, c in enumerate(colors):
        draw_rect(i * 52.5, 0, 52.5, 2, *c)

    # Title
    pdf.set_xy(0, 14)
    pdf.set_font('Helvetica', 'B', 26)
    text_color(*C_WHITE)
    pdf.cell(0, 12, 'BEAT THE TABLE', align='C', ln=True)

    pdf.set_font('Helvetica', '', 9)
    text_color(*C_MUTED)
    pdf.cell(0, 6, 'AI-Powered Football Tactical Intelligence Report', align='C', ln=True)

    # Date
    pdf.set_font('Helvetica', '', 8)
    pdf.cell(0, 5, f'Generated: {datetime.now().strftime("%d %B %Y, %H:%M")}', align='C', ln=True)

    # Vs card
    pdf.ln(8)
    card_y = pdf.get_y()
    draw_rect(25, card_y, 160, 28, *C_ELEVATED)
    # Blue left accent
    draw_rect(25, card_y, 3, 28, *C_BLUE)
    # Red right accent
    draw_rect(182, card_y, 3, 28, *C_RED)

    pdf.set_xy(28, card_y + 5)
    pdf.set_font('Helvetica', 'B', 18)
    text_color(*C_BLUE)
    pdf.cell(70, 10, team_a, align='C')

    pdf.set_font('Helvetica', 'B', 11)
    text_color(*C_MUTED)
    pdf.cell(20, 10, 'VS', align='C')

    pdf.set_font('Helvetica', 'B', 18)
    text_color(*C_RED)
    pdf.cell(70, 10, team_b, align='C', ln=True)

    pdf.set_xy(28, card_y + 17)
    pdf.set_font('Helvetica', '', 7)
    text_color(*C_MUTED)
    pdf.cell(70, 5, f'Formation: {formation_a}', align='C')
    pdf.cell(20, 5, '')
    pdf.cell(70, 5, f'Formation: {formation_b}', align='C', ln=True)

    # Opponent style badge
    pdf.ln(4)
    pdf.set_font('Helvetica', '', 8)
    text_color(*C_MUTED)
    pdf.cell(0, 5, f'Opponent Playstyle:  {opp_style}', align='C', ln=True)

    # ── WIN PROBABILITY SECTION ──────────────────────────────────────────────
    pdf.ln(4)
    section_header('WIN PROBABILITY ANALYSIS', '📊')

    win_pct  = probs['Win']
    draw_pct = probs['Draw']
    loss_pct = probs['Loss']

    prob_bar(f'{team_a} Win',  win_pct,  *C_BLUE)
    prob_bar('Draw',           draw_pct, *C_MUTED)
    prob_bar(f'{team_b} Win',  loss_pct, *C_RED)

    # Verdict box
    if win_pct > loss_pct + 0.1:
        verdict_team, verdict_color = team_a, C_BLUE
    elif loss_pct > win_pct + 0.1:
        verdict_team, verdict_color = team_b, C_RED
    else:
        verdict_team, verdict_color = 'Too Close to Call', C_AMBER

    y_v = pdf.get_y() + 2
    draw_rect(10, y_v, 190, 10, *C_SURFACE)
    draw_rect(10, y_v, 3, 10, *verdict_color)
    pdf.set_xy(17, y_v + 2.5)
    pdf.set_font('Helvetica', 'B', 9)
    text_color(*verdict_color)
    pdf.cell(0, 5, f'Model Verdict: {verdict_team}  {max(win_pct, loss_pct):.1%}  probability of winning')
    pdf.ln(14)
    text_color(*C_TEXT)

    # ── AI COMMENTARY ────────────────────────────────────────────────────────
    section_header('AI MATCH PREVIEW', '🎤')
    commentary = explanation.get('explanation_text', 'No commentary available.')
    y_c = pdf.get_y()
    draw_rect(10, y_c, 190, 2, *C_BLUE)
    pdf.ln(5)
    pdf.set_xy(12, pdf.get_y())
    pdf.set_font('Helvetica', '', 8.5)
    text_color(*C_TEXT)
    pdf.multi_cell(186, 5, commentary)
    pdf.ln(4)

    # ─────────────────────────────────────────────────────────────────────────
    # PAGE 2: STARTING XIs
    # ─────────────────────────────────────────────────────────────────────────
    pdf.add_page()
    set_bg(*C_BG)
    pdf.rect(0, 0, 210, 297, 'F')

    section_header('RECOMMENDED STARTING XIs', '📋')

    # Save pitch PNGs as temp files (fpdf2 needs path or bytes-like for images)
    import tempfile, os

    tmp_a = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
    tmp_a.write(pitch_a_bytes)
    tmp_a.flush()
    tmp_a.close()

    tmp_b = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
    tmp_b.write(pitch_b_bytes)
    tmp_b.flush()
    tmp_b.close()

    try:
        # Pitches side by side
        pdf.set_font('Helvetica', 'B', 9)
        text_color(*C_BLUE)
        pdf.set_x(12)
        pdf.cell(89, 6, f'{team_a}  ({formation_a})', align='C')
        text_color(*C_RED)
        pdf.cell(89, 6, f'{team_b}  ({formation_b})', align='C', ln=True)

        pitch_y = pdf.get_y()
        pdf.image(tmp_a.name, x=12, y=pitch_y, w=89)
        pdf.image(tmp_b.name, x=109, y=pitch_y, w=89)
        pdf.set_y(pitch_y + 108)  # approx height of 4:5.6 figure at w=89
    finally:
        os.unlink(tmp_a.name)
        os.unlink(tmp_b.name)

    # Player tables
    pdf.ln(4)
    col_w = [14, 38, 16, 16, 96]

    def player_table_header(x_start, team_color):
        pdf.set_xy(x_start, pdf.get_y())
        draw_rect(x_start, pdf.get_y(), 180, 6, *C_ELEVATED)
        pdf.set_font('Helvetica', 'B', 6.5)
        text_color(*C_MUTED)
        for label, w in zip(['POS', 'PLAYER', 'OVR', 'TACTICAL', 'COACH NOTE'], col_w):
            pdf.cell(w, 6, label)
        pdf.ln(6)

    def player_row(p, x_start, tactical_color):
        row_y = pdf.get_y()
        # Alternate row bg
        draw_rect(x_start, row_y, 180, 5.5, *(C_SURFACE if starting_xi_a.index(p) % 2 == 0 else C_BG) if p in starting_xi_a else C_SURFACE)

        # Position badge
        pc = POS_COLORS.get(p['role'], C_MUTED)
        draw_rect(x_start, row_y + 0.8, 13, 4, *pc)
        pdf.set_xy(x_start, row_y + 0.8)
        pdf.set_font('Helvetica', 'B', 6)
        text_color(*C_WHITE)
        pdf.cell(13, 4, p['role'], align='C')

        # Player name
        pdf.set_xy(x_start + 14, row_y + 0.8)
        pdf.set_font('Helvetica', 'B', 7)
        text_color(*C_TEXT)
        pdf.cell(37, 4, p['name'][:22])

        # OVR
        pdf.set_xy(x_start + 52, row_y + 0.8)
        pdf.set_font('Helvetica', '', 7)
        text_color(*C_MUTED)
        pdf.cell(15, 4, str(p.get('base_rating', '')))

        # Tactical rating
        pdf.set_xy(x_start + 67, row_y + 0.8)
        pdf.set_font('Helvetica', 'B', 7)
        text_color(*tactical_color)
        pdf.cell(15, 4, str(p.get('adjusted_rating', '')))

        # Coach note (truncated)
        note = p.get('rationale', '')[:60]
        pdf.set_xy(x_start + 82, row_y + 0.8)
        pdf.set_font('Helvetica', '', 5.5)
        text_color(*C_MUTED)
        pdf.cell(0, 4, note)

        pdf.ln(5.5)
        text_color(*C_TEXT)

    pdf.set_font('Helvetica', 'B', 9)
    text_color(*C_BLUE)
    pdf.cell(0, 6, f'{team_a} — Tactical Rationale', ln=True)
    player_table_header(10, C_BLUE)
    for p in starting_xi_a:
        player_row(p, 10, C_BLUE)

    pdf.ln(4)
    pdf.set_font('Helvetica', 'B', 9)
    text_color(*C_RED)
    pdf.cell(0, 6, f'{team_b} — Tactical Rationale', ln=True)
    player_table_header(10, C_RED)
    for p in starting_xi_b:
        player_row(p, 10, C_RED)

    # ─────────────────────────────────────────────────────────────────────────
    # PAGE 3: WEAK LINKS + TOURNAMENT
    # ─────────────────────────────────────────────────────────────────────────
    pdf.add_page()
    set_bg(*C_BG)
    pdf.rect(0, 0, 210, 297, 'F')

    # ── WEAK LINK DETECTOR ───────────────────────────────────────────────────
    section_header('WEAK LINK DETECTOR', '🔍')
    pdf.set_font('Helvetica', '', 8)
    text_color(*C_MUTED)
    pdf.multi_cell(0, 4.5,
        f"Our AI scanned {team_b}'s squad and identified the three biggest "
        f"player-vs-player mismatches in {team_a}'s lineup.")
    pdf.ln(3)

    sev_map = {'🔴 Critical': C_RED, '🟠 Warning': (234, 88, 12), '🟡 Caution': C_AMBER, '🟢 Solid': C_GREEN}

    for i, wl in enumerate(weak_links, 1):
        sc = sev_map.get(wl.get('severity', ''), C_MUTED)
        card_y = pdf.get_y()
        card_h = 28

        draw_rect(10, card_y, 190, card_h, *C_ELEVATED)
        draw_rect(10, card_y, 3, card_h, *sc)

        # Rank + severity
        pdf.set_xy(15, card_y + 3)
        pdf.set_font('Helvetica', 'B', 7)
        text_color(*C_MUTED)
        pdf.cell(25, 4, f'MISMATCH #{i}')

        # Severity badge
        sev_label = wl.get('severity', 'Caution').replace('🔴 ', '').replace('🟠 ', '').replace('🟡 ', '').replace('🟢 ', '')
        draw_rect(155, card_y + 2, 40, 5, *sc)
        pdf.set_xy(155, card_y + 2.8)
        pdf.set_font('Helvetica', 'B', 6.5)
        text_color(*C_WHITE)
        pdf.cell(40, 3.5, sev_label, align='C')

        # Names
        pdf.set_xy(15, card_y + 9)
        pdf.set_font('Helvetica', 'B', 10)
        text_color(*C_WHITE)
        pdf.cell(70, 5, wl.get('my_player', ''))
        pdf.set_font('Helvetica', '', 9)
        text_color(*C_MUTED)
        pdf.cell(12, 5, 'vs', align='C')
        pdf.set_font('Helvetica', 'B', 10)
        text_color(*C_WHITE)
        pdf.cell(0, 5, wl.get('threat_player', ''), ln=True)

        # Stat pills
        pdf.set_xy(15, card_y + 16)
        pdf.set_font('Helvetica', '', 7)
        text_color(*C_BLUE)
        my_stat = f"My {wl.get('my_key_stat','').capitalize()} ({wl.get('my_role','')}): {wl.get('my_stat_val','')}"
        pdf.cell(75, 4, my_stat)
        text_color(*C_RED)
        their_stat = f"Their {wl.get('threat_key_stat','').capitalize()} ({wl.get('threat_role','')}): {wl.get('threat_stat_val','')}"
        pdf.cell(75, 4, their_stat)
        text_color(*sc)
        pdf.cell(0, 4, f"Gap: {wl.get('vulnerability_score', 0):+d}", ln=True)

        # Description
        desc = wl.get('description', '')[:110]
        pdf.set_xy(15, card_y + 22)
        pdf.set_font('Helvetica', 'I', 6.5)
        text_color(*C_MUTED)
        pdf.cell(0, 3.5, desc)

        pdf.ln(card_h + 2)
        text_color(*C_TEXT)

    # ── TOURNAMENT PROJECTION ───────────────────────────────────────────────
    pdf.ln(2)
    section_header('TOURNAMENT PROJECTION  (Monte Carlo · 1,000 runs)', '🏆')

    table = sim_res.get('table', [])
    stage_icons = {'Group Stage': '🏁', 'R16': '⚡', 'QF': '🔥', 'SF': '💥', 'Final': '🏆', 'Winner': '🥇'}
    headers = ['Stage', 'Advance %', 'Top Opponent', 'Meet %']
    col_ws  = [42, 38, 64, 38]

    # Header row
    draw_rect(10, pdf.get_y(), 190, 7, *C_ELEVATED)
    pdf.set_xy(12, pdf.get_y() + 1.5)
    pdf.set_font('Helvetica', 'B', 7)
    text_color(*C_MUTED)
    for h, w in zip(headers, col_ws):
        pdf.cell(w, 4, h.upper())
    pdf.ln(7)

    for row in table:
        adv = row.get('Advancement Probability', 0)
        opp_freq = row.get('Opponent Frequency', 0)
        stage = row.get('Stage', '')
        icon = stage_icons.get(stage, '')
        is_hardest = (stage == sim_res.get('hardest_stage', ''))

        row_y = pdf.get_y()
        row_bg = (30, 41, 59) if table.index(row) % 2 == 0 else C_BG
        draw_rect(10, row_y, 190, 7, *row_bg)
        if is_hardest:
            draw_rect(10, row_y, 2, 7, *C_AMBER)

        # Stage
        pdf.set_xy(12, row_y + 1.5)
        pdf.set_font('Helvetica', 'B', 8)
        pct = adv
        color = C_GREEN if pct > 0.5 else C_AMBER if pct > 0.25 else C_RED
        text_color(*color)
        pdf.cell(col_ws[0], 4, f'{icon} {stage}')

        # Advance % + bar
        pdf.set_font('Helvetica', 'B', 8)
        text_color(*color)
        pdf.cell(14, 4, f'{pct:.0%}')
        bar_x = pdf.get_x()
        draw_rect(bar_x, row_y + 3, 22, 1.5, *C_ELEVATED)
        draw_rect(bar_x, row_y + 3, min(22, pct * 22), 1.5, *color)
        pdf.cell(24, 4, '')

        # Top opponent
        pdf.set_font('Helvetica', '', 7.5)
        text_color(*C_MUTED)
        pdf.cell(col_ws[2], 4, row.get('Most Common Opponent', 'N/A'))

        # Meet %
        pdf.set_font('Helvetica', 'B', 7.5)
        text_color(*C_TEXT)
        pdf.cell(col_ws[3], 4, f'{opp_freq:.0%}', ln=True)

        text_color(*C_TEXT)

    # Hardest stage callout
    if sim_res.get('hardest_stage'):
        pdf.ln(4)
        y_hs = pdf.get_y()
        draw_rect(10, y_hs, 190, 10, 25, 20, 5)
        draw_rect(10, y_hs, 3, 10, *C_AMBER)
        pdf.set_xy(16, y_hs + 2.5)
        pdf.set_font('Helvetica', 'B', 8.5)
        text_color(*C_AMBER)
        drop = sim_res.get('hardest_stage_drop', 0)
        pdf.cell(0, 5,
            f"Toughest Round for {team_a}:  {sim_res['hardest_stage']}  "
            f"(probability drops {drop:.1%} here)")
        text_color(*C_TEXT)

    # ── FOOTER ──────────────────────────────────────────────────────────────
    pdf.set_y(-20)
    draw_rect(0, pdf.get_y(), 210, 20, *C_SURFACE)
    pdf.set_font('Helvetica', '', 7)
    text_color(*C_MUTED)
    pdf.cell(0, 10,
        f'Beat the Table  ·  AI Football Analytics  ·  Generated {datetime.now().strftime("%d %B %Y")}',
        align='C')

    return bytes(pdf.output())
