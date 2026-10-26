import {
  Button as NativeButton,
  TextField,
  Toggle,
  type TextFieldRef,
} from '@expo/ui/swift-ui';
import { foregroundStyle } from '@expo/ui/swift-ui/modifiers';
import * as Haptics from 'expo-haptics';
import { useLocalSearchParams, useRouter } from 'expo-router';
import { useSQLiteContext } from 'expo-sqlite';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Alert } from 'react-native';

import {
  addCourse,
  deleteCourse,
  readStructure,
  setCourseRequirements,
  updateCourse,
  type AcademicStructure,
} from '../../src/database/academicRepo';
import { getCourse, type CourseDetail } from '../../src/database/repo';
import { LETTERS, LEVELS } from '../../src/domain/gpa';
import { NativeScreen, NativeText, Section } from '../../src/ui/native';
import { useTheme } from '../../src/ui/useTheme';

type Status = 'completed' | 'current' | 'planned';

const STATUSES: { value: Status; label: string }[] = [
  { value: 'completed', label: 'Completed' },
  { value: 'current', label: 'Current' },
  { value: 'planned', label: 'Planned' },
];

/**
 * Add or edit one course.
 *
 * A sheet, not a screen: the grid is the destination and this is the detent. An
 * imported value that was changed keeps its original in the notice at the top, so a
 * correction never quietly rewrites what the transcript said.
 *
 * The fields are native text fields. They hold their own text, so each one is
 * prefilled through its ref once the course is read, and every keystroke is reported
 * back into React state — which is what the validation and the save payload read.
 */
export default function CourseEditor() {
  const db = useSQLiteContext();
  const router = useRouter();
  const { palette } = useTheme();
  const params = useLocalSearchParams<{
    id?: string;
    mode?: string;
    scheduleId?: string;
    yearId?: string;
    termId?: string;
  }>();
  const editingId = typeof params.id === 'string' && params.id ? params.id : null;

  const [structure, setStructure] = useState<AcademicStructure | null>(null);
  const [detail, setDetail] = useState<CourseDetail | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const [title, setTitle] = useState('');
  const [code, setCode] = useState('');
  const [credits, setCredits] = useState('1');
  const [level, setLevel] = useState('regular');
  const [status, setStatus] = useState<Status>('planned');
  const [mark, setMark] = useState('');
  const [expected, setExpected] = useState<string>('A');
  const [includeInGpa, setIncludeInGpa] = useState(true);
  const [requirementIds, setRequirementIds] = useState<string[]>([]);

  const titleRef = useRef<TextFieldRef>(null);
  const codeRef = useRef<TextFieldRef>(null);
  const creditsRef = useRef<TextFieldRef>(null);
  const markRef = useRef<TextFieldRef>(null);

  const load = useCallback(async () => {
    try {
      const nextStructure = await readStructure(db);
      setStructure(nextStructure);
      if (!editingId) {
        void creditsRef.current?.setText('1');
        setError(null);
        return;
      }
      const next = await getCourse(db, editingId);
      if (!next) {
        setError('That course is no longer stored.');
        return;
      }
      setDetail(next);
      const course = next.course;
      const nextTitle = course.title ?? '';
      const nextCode = course.code ?? '';
      const nextCredits = course.credits != null ? String(course.credits) : '1';
      const nextMark = course.grade ?? '';
      setTitle(nextTitle);
      setCode(nextCode);
      setCredits(nextCredits);
      setLevel(course.level ?? 'regular');
      setStatus(
        course.status === 'completed' || course.status === 'current' || course.status === 'planned'
          ? course.status
          : 'completed',
      );
      setMark(nextMark);
      setExpected(course.expected_grade ?? 'A');
      setIncludeInGpa(course.include_in_gpa !== 0);
      setRequirementIds(
        Object.entries(nextStructure.links)
          .filter(([, ids]) => ids.includes(editingId))
          .map(([requirementId]) => requirementId),
      );
      // The fields were mounted with the sheet, so their refs are available here.
      void titleRef.current?.setText(nextTitle);
      void codeRef.current?.setText(nextCode);
      void creditsRef.current?.setText(nextCredits);
      void markRef.current?.setText(nextMark);
      setError(null);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Pace could not open this course.');
    }
  }, [db, editingId]);

  useEffect(() => {
    void load();
  }, [load]);

  const course = detail?.course ?? null;
  const creditsNumber = Number(credits.replace(',', '.'));
  const canSave = useMemo(() => {
    if (!title.trim()) return false;
    if (!Number.isFinite(creditsNumber) || creditsNumber <= 0) return false;
    if (!editingId && !(params.scheduleId && params.yearId)) return false;
    return true;
  }, [title, creditsNumber, editingId, params.scheduleId, params.yearId]);

  const toggleRequirement = (id: string) => {
    setRequirementIds((current) =>
      current.includes(id) ? current.filter((row) => row !== id) : [...current, id],
    );
  };

  const save = async () => {
    if (!canSave || busy) return;
    setBusy(true);
    try {
      let courseId = editingId;
      if (editingId) {
        await updateCourse(db, editingId, {
          title: title.trim(),
          code: code.trim() || null,
          credits: creditsNumber,
          level,
          status,
          grade: status === 'completed' ? mark.trim() || null : null,
          expectedGrade: status === 'completed' ? null : expected,
          includeInGpa,
        });
      } else {
        courseId = await addCourse(db, {
          title: title.trim(),
          credits: creditsNumber,
          level,
          expectedGrade: status === 'completed' ? null : expected,
          grade: status === 'completed' ? mark.trim() || null : null,
          status,
          academicYearId: params.yearId as string,
          scheduleId: params.scheduleId as string,
          academicTermId: params.termId ?? null,
        });
      }
      if (courseId) await setCourseRequirements(db, courseId, requirementIds);
      Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success).catch(() => {});
      router.back();
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Pace could not save this course.');
      setBusy(false);
    }
  };

  const remove = () => {
    if (!editingId) return;
    Alert.alert('Delete course?', title.trim() || 'This course', [
      { text: 'Cancel', style: 'cancel' },
      {
        text: 'Delete',
        style: 'destructive',
        onPress: () => {
          deleteCourse(db, editingId)
            .then(() => {
              Haptics.notificationAsync(Haptics.NotificationFeedbackType.Warning).catch(() => {});
              router.back();
            })
            .catch((cause: unknown) =>
              setError(cause instanceof Error ? cause.message : 'Pace could not delete this course.'),
            );
        },
      },
    ]);
  };

  const editOriginal =
    course && course.edited === 1 && course.original_grade && course.original_grade !== mark.trim()
      ? course.original_grade
      : null;

  return (
    <NativeScreen
      topInset={false}
      title={editingId ? 'Edit Course' : 'New Course'}
      onBack={() => router.back()}
      primary={{
        label: busy ? 'Saving…' : 'Save',
        disabled: !canSave || busy,
        onPress: () => void save(),
      }}
    >
      {error ? (
        <Section>
          <NativeText modifiers={[foregroundStyle(palette.danger)]}>{error}</NativeText>
        </Section>
      ) : null}

      {editOriginal ? (
        <Section title="Transcript" footer={<NativeText>{`The printed mark was ${editOriginal}.`}</NativeText>}>
          <NativeText>Edited</NativeText>
        </Section>
      ) : null}

      <Section title="Title">
        <TextField
          ref={titleRef}
          placeholder="Course name"
          autoFocus={!editingId}
          onTextChange={setTitle}
        />
      </Section>

      <Section title="Code">
        <TextField ref={codeRef} placeholder="Optional" onTextChange={setCode} />
      </Section>

      <Section title="Credits">
        <TextField ref={creditsRef} placeholder="1" onTextChange={setCredits} />
      </Section>

      <Section title="Level">
        {LEVELS.map((row) => (
          <NativeButton
            key={row.value}
            label={row.label}
            onPress={() => setLevel(row.value)}
            systemImage={level === row.value ? 'checkmark' : undefined}
          />
        ))}
      </Section>

      <Section title="Status">
        {STATUSES.map((row) => (
          <NativeButton
            key={row.value}
            label={row.label}
            onPress={() => setStatus(row.value)}
            systemImage={status === row.value ? 'checkmark' : undefined}
          />
        ))}
      </Section>

      {status === 'completed' ? (
        <Section title="Final mark">
          <TextField ref={markRef} placeholder="94 or A" onTextChange={setMark} />
        </Section>
      ) : (
        <Section title="Expected grade">
          {LETTERS.map((letter) => (
            <NativeButton
              key={letter}
              label={letter}
              onPress={() => setExpected(letter)}
              systemImage={expected === letter ? 'checkmark' : undefined}
            />
          ))}
        </Section>
      )}

      <Section>
        <Toggle label="Include in GPA" isOn={includeInGpa} onIsOnChange={setIncludeInGpa} />
      </Section>

      {structure && structure.requirements.length ? (
        <Section title="Requirements">
          {structure.requirements.map((requirement) => (
            <NativeButton
              key={requirement.id}
              label={requirement.name}
              onPress={() => toggleRequirement(requirement.id)}
              systemImage={requirementIds.includes(requirement.id) ? 'checkmark' : undefined}
            />
          ))}
        </Section>
      ) : null}

      {editingId ? (
        <Section>
          <NativeButton label="Delete Course" role="destructive" onPress={remove} />
        </Section>
      ) : null}
    </NativeScreen>
  );
}
