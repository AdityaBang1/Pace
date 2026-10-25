from pathlib import Path
import csv
from pace.real_data import load_usc_examples, split_real_groups, exclude_intrinsically_ambiguous


def test_real_catalog_loader_preserves_cell_text_and_subject_splits(tmp_path: Path):
    source = tmp_path / 'tiny.csv'
    fields = ['Program Code','Class ID','Class Name','Catalogue','Units','Prerequisites','Restrictions','Time','Days','Class Type','Instructor']
    with source.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=fields); writer.writeheader()
        writer.writerow({'Program Code':'BIO','Class ID':'BIO 101: Intro','Class Name':'Intro Biology','Catalogue':'Study of living systems.','Units':'(4.0 units)','Prerequisites':'Not available','Restrictions':'Not available','Time':'10:00am','Days':'Mon','Class Type':'Lecture','Instructor':'Person Name'})
        writer.writerow({'Program Code':'CHEM','Class ID':'CHEM 201: Chemistry','Class Name':'Organic Chemistry','Catalogue':'Study reaction mechanisms.','Units':'(3.0 units)','Prerequisites':'BIO 101','Restrictions':'Not available','Time':'2:00pm','Days':'Tue','Class Type':'Lecture','Instructor':'Other Person'})
        writer.writerow({'Program Code':'MATH','Class ID':'MATH 130: Algebra','Class Name':'College Algebra','Catalogue':'A study of algebraic functions.','Units':'(4.0 units)','Prerequisites':'Not available','Restrictions':'Not available','Time':'9:00am','Days':'Wed','Class Type':'Lecture','Instructor':'Third Person'})
    rows=load_usc_examples(source)
    assert {row['label'] for row in rows} == {'course_title','course_description','credit_units','meeting_time','meeting_days','class_type','prerequisites'}
    assert all('Person Name' not in row['text'] and 'Other Person' not in row['text'] for row in rows)
    assert any(row['text']=='Study of living systems.' for row in rows)
    clean,ambiguous,excluded=exclude_intrinsically_ambiguous(rows)
    splits=split_real_groups(clean,seed=1)
    assert not ({r['group_id'] for r in splits['train']} & {r['group_id'] for r in splits['test']})


def test_conflicting_literal_values_are_excluded_from_text_only_task():
    rows=[{'text':'TBA','label':'meeting_days','group_id':'s1'},
          {'text':' tba  ','label':'meeting_time','group_id':'s2'},
          {'text':'Monday','label':'meeting_days','group_id':'s1'}]
    clean,ambiguous,excluded=exclude_intrinsically_ambiguous(rows)
    assert ambiguous=={'tba':['meeting_days','meeting_time']}
    assert [x['text'] for x in clean]==['Monday']
    assert len(excluded)==2
