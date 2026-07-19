# Master's Thesis Presentation (Gate V1.2.1)

Results-first, fully editable PowerPoint deck for the master's thesis presentation.

## Files

- `editable/master_thesis_presentation_v1.pptx`: 16 main slides and 7 backup slides.
- `presentation_plan.md`: results-first narrative and slide map.
- `speaker_notes_bilingual.md`: English and Japanese notes for all 23 slides.
- `PRESENTATION_MANIFEST.tsv`: slide relationships, object types, and sources.
- `../../../scripts/generate_thesis_presentation.py`: deterministic generator and validator.

## Presentation timing

Presentation timing is intentionally not fixed.
The user will adjust slide selection and speaking time after rehearsal.

Every notes entry uses `Duration: [USER TO EDIT]` in the PowerPoint notes pane.

## Correctness role

Correctness is treated as required validation. It is not presented as the main research result.
Detailed correctness evidence and the historical malformed input remain in Backup Slides 22–23.
The Tier A/B data and all 13 comparisons are unchanged.

## Editability

- Editable objects: 291
- Native charts: 6
- Native tables: 5
- Raster pictures: 0
- Raster-only slides: 0

All diagrams use native PowerPoint shapes and connectors. Charts retain embedded workbooks.

## Regeneration

```bash
cd thesis_bc_project
PYTHONPATH=<deps> python3 scripts/generate_thesis_presentation.py
```

The generator checks canonical measured values, slide language, font floors, text fit,
object bounds, editable objects, notes, chart identifiers, and raster use.
