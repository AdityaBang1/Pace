import * as Haptics from 'expo-haptics';
import { useLocalSearchParams, useRouter } from 'expo-router';
import { useSQLiteContext } from 'expo-sqlite';
import { useCallback, useEffect, useMemo, useState, type ReactNode } from 'react';
import { Alert, ScrollView, Switch, Text, TextInput, View } from 'react-native';

import {
  addCourse,
  deleteCourse,
  readStructure,
  setCourseRequirements,
  updateCourse,
  type AcademicStructure,
} from '../../src/database/academicRepo';
import { getCourse, type CourseDetail } from '../../src/database/repo';
import { LETTERS, LEVELS, levelLabel } from '../../src/domain/gpa';
import { Button, Chip, FieldRow, Notice, PageHeader, SectionHeader } from '../../src/ui/components';
import { space, type } from '../../src/ui/theme';
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

  const load = useCallback(async () => {
    try {
      const nextStructure = await readStructure(db);
      setStructure(nextStructure);
      if (!editingId) {
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
      setTitle(course.title ?? '');
      setCode(course.code ?? '');
      setCredits(course.credits != null ? String(course.credits) : '1');
      setLevel(course.level ?? 'regular');
      setStatus(
        course.status === 'completed' || course.status === 'current' || course.status === 'planned'
          ? course.status
          : 'completed',
      );
      setMark(course.grade ?? '');
      setExpected(course.expected_grade ?? 'A');
      setIncludeInGpa(course.include_in_gpa !== 0);
      setRequirementIds(
        Object.entries(nextStructure.links)
          .filter(([, ids]) => ids.includes(editingId))
          .map(([requirementId]) => requirementId),
      );
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
    <View style={{ flex: 1, backgroundColor: palette.background }}>
      <PageHeader
        title={editingId ? 'Edit Course' : 'New Course'}
        onBack={() => router.back()}
        right={
          <Button title="Save" compact disabled={!canSave || busy} loading={busy} onPress={() => void save()} />
        }
      />
      <ScrollView
        keyboardShouldPersistTaps="handled"
        contentContainerStyle={{ paddingHorizontal: space.xl, paddingBottom: space.xxxl, gap: space.md }}
      >
        {error ? <Notice tone="danger" body={error} /> : null}
        {editOriginal ? (
          <Notice tone="warn" body={`Transcript said ${editOriginal}.`} />
        ) : null}

        <Field label="Title">
          <TextInput
            value={title}
            onChangeText={setTitle}
            placeholder="Course name"
            placeholderTextColor={palette.inkFaint}
            autoFocus={!editingId}
            style={[type.body, inputStyle(palette.ink, palette.line, palette.surfaceSunken)]}
            accessibilityLabel="Course title"
          />
        </Field>

        <Field label="Code">
          <TextInput
            value={code}
            onChangeText={setCode}
            placeholder="Optional"
            placeholderTextColor={palette.inkFaint}
            autoCapitalize="characters"
            style={[type.body, inputStyle(palette.ink, palette.line, palette.surfaceSunken)]}
            accessibilityLabel="Course code"
          />
        </Field>

        <Field label="Credits">
          <TextInput
            value={credits}
            onChangeText={setCredits}
            keyboardType="decimal-pad"
            style={[type.body, inputStyle(palette.ink, palette.line, palette.surfaceSunken)]}
            accessibilityLabel="Credits"
          />
        </Field>

        <Field label="Level">
          <View style={{ flexDirection: 'row', flexWrap: 'wrap', gap: space.sm }}>
            {LEVELS.map((row) => (
              <Chip
                key={row.value}
                label={row.label}
                selected={level === row.value}
                tone={level === row.value ? 'accent' : 'neutral'}
                onPress={() => setLevel(row.value)}
              />
            ))}
          </View>
          <Text style={[type.caption, { color: palette.inkFaint, marginTop: space.xs }]}>
            {levelLabel(level as 'regular' | 'honors' | 'ap')}
          </Text>
        </Field>

        <Field label="Status">
          <View style={{ flexDirection: 'row', flexWrap: 'wrap', gap: space.sm }}>
            {STATUSES.map((row) => (
              <Chip
                key={row.value}
                label={row.label}
                selected={status === row.value}
                tone={status === row.value ? 'accent' : 'neutral'}
                onPress={() => setStatus(row.value)}
              />
            ))}
          </View>
        </Field>

        {status === 'completed' ? (
          <Field label="Final Mark">
            <TextInput
              value={mark}
              onChangeText={setMark}
              placeholder="94 or A"
              placeholderTextColor={palette.inkFaint}
              autoCapitalize="characters"
              style={[type.body, inputStyle(palette.ink, palette.line, palette.surfaceSunken)]}
              accessibilityLabel="Final mark"
            />
          </Field>
        ) : (
          <Field label="Expected Grade">
            <View style={{ flexDirection: 'row', flexWrap: 'wrap', gap: space.sm }}>
              {LETTERS.map((letter) => (
                <Chip
                  key={letter}
                  label={letter}
                  selected={expected === letter}
                  tone={expected === letter ? 'accent' : 'neutral'}
                  onPress={() => setExpected(letter)}
                />
              ))}
            </View>
          </Field>
        )}

        <FieldRow
          label="Include in GPA"
          value={
            <Switch
              value={includeInGpa}
              onValueChange={setIncludeInGpa}
              accessibilityLabel="Include this course in GPA"
            />
          }
        />

        {structure && structure.requirements.length ? (
          <>
            <SectionHeader>Requirements</SectionHeader>
            <View style={{ flexDirection: 'row', flexWrap: 'wrap', gap: space.sm }}>
              {structure.requirements.map((requirement) => (
                <Chip
                  key={requirement.id}
                  label={requirement.name}
                  selected={requirementIds.includes(requirement.id)}
                  tone={requirementIds.includes(requirement.id) ? 'accent' : 'neutral'}
                  onPress={() => toggleRequirement(requirement.id)}
                />
              ))}
            </View>
          </>
        ) : null}

        {editingId ? (
          <View style={{ marginTop: space.xl }}>
            <Button title="Delete Course" kind="quiet" onPress={remove} />
          </View>
        ) : null}
      </ScrollView>
    </View>
  );
}

function Field({ label, children }: { label: string; children: ReactNode }) {
  const { palette } = useTheme();
  return (
    <View style={{ gap: space.sm, marginTop: space.sm }}>
      <Text style={[type.label, { color: palette.inkFaint }]}>{label}</Text>
      {children}
    </View>
  );
}

function inputStyle(color: string, border: string, background: string) {
  return {
    color,
    borderColor: border,
    borderWidth: 1,
    backgroundColor: background,
    borderRadius: 12,
    paddingHorizontal: space.md,
    paddingVertical: space.md,
  } as const;
}
