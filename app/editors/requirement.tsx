import * as Haptics from 'expo-haptics';
import { useLocalSearchParams, useRouter } from 'expo-router';
import { useSQLiteContext } from 'expo-sqlite';
import { useEffect, useRef, useState } from 'react';
import { Alert } from 'react-native';

import {
  Button as NativeButton,
  Section,
  Text as NativeText,
  TextField,
  type TextFieldRef,
} from '@expo/ui/swift-ui';
import { disabled } from '@expo/ui/swift-ui/modifiers';

import {
  deleteRequirement,
  ensureStructure,
  readStructure,
  saveRequirement,
  setCourseRequirements,
  type AcademicStructure,
} from '../../src/database/academicRepo';
import { listCourses, type CourseRow } from '../../src/database/repo';
import { statusOf } from '../../src/domain/record';
import { NativeScreen } from '../../src/ui/native';

/**
 * One requirement: a name, a credit goal, and the courses that count towards it.
 *
 * Mapping is explicit. Pace never decides that "AP Human Geography" is a Social
 * Studies credit because a pattern in its title said so; the student taps it.
 */
export default function RequirementEditor() {
  const { id } = useLocalSearchParams<{ id?: string }>();
  const db = useSQLiteContext();
  const router = useRouter();

  const [structure, setStructure] = useState<AcademicStructure | null>(null);
  const [courses, setCourses] = useState<CourseRow[]>([]);
  const [originalLinks, setOriginalLinks] = useState<Record<string, string[]>>({});
  const [name, setName] = useState('');
  const [credits, setCredits] = useState('');
  const [selected, setSelected] = useState<string[]>([]);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [hydrated, setHydrated] = useState(false);

  const nameRef = useRef<TextFieldRef>(null);
  const creditsRef = useRef<TextFieldRef>(null);

  useEffect(() => {
    void (async () => {
      await ensureStructure(db);
      const [next, rows] = await Promise.all([readStructure(db), listCourses(db)]);
      setStructure(next);
      setCourses(rows);
      setOriginalLinks(next.links);
      if (id) {
        const requirement = next.requirements.find((row) => row.id === id);
        if (requirement) {
          setName(requirement.name);
          setCredits(String(requirement.required_credits));
        }
        setSelected(next.links[id] ?? []);
      }
      setHydrated(true);
    })();
  }, [db, id]);

  // A SwiftUI field owns its text, so the stored values are pushed in once.
  useEffect(() => {
    if (!hydrated) return;
    void nameRef.current?.setText(name);
    void creditsRef.current?.setText(credits);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [hydrated]);

  const value = Number(credits.replace(',', '.'));
  const valid = name.trim().length > 0 && Number.isFinite(value) && value > 0;

  const toggle = (courseId: string) => {
    setSelected((current) =>
      current.includes(courseId) ? current.filter((row) => row !== courseId) : [...current, courseId],
    );
  };

  const save = async () => {
    if (!valid) return;
    setSaving(true);
    try {
      const requirementId = await saveRequirement(db, {
        id,
        name: name.trim(),
        requiredCredits: Math.round(value * 100) / 100,
      });
      // Every course whose membership changed is written, so a course can belong to
      // more than one requirement without either edit erasing the other.
      for (const course of courses) {
        const previous = (originalLinks[requirementId] ?? []).includes(course.id);
        const now = selected.includes(course.id);
        if (previous === now) continue;
        const others = Object.entries(originalLinks)
          .filter(([requirement]) => requirement !== requirementId)
          .filter(([, ids]) => ids.includes(course.id))
          .map(([requirement]) => requirement);
        await setCourseRequirements(db, course.id, now ? [...others, requirementId] : others);
      }
      Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success).catch(() => {});
      router.back();
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'The requirement could not be saved.');
      setSaving(false);
    }
  };

  const remove = () => {
    if (!id) return;
    Alert.alert('Delete requirement?', name || 'This requirement', [
      { text: 'Cancel', style: 'cancel' },
      {
        text: 'Delete',
        style: 'destructive',
        onPress: () => {
          void (async () => {
            await deleteRequirement(db, id);
            router.back();
          })();
        },
      },
    ]);
  };

  const grouped = (structure?.years ?? [])
    .map((year) => ({
      year,
      rows: courses.filter(
        (course) => course.academic_year_id === year.id && statusOf(course) !== 'planned',
      ),
    }))
    .filter((group) => group.rows.length > 0);

  return (
    <NativeScreen
      topInset={false}
      title={id ? 'Edit Requirement' : 'New Requirement'}
      onBack={() => router.back()}
    >
      {error ? (
        <Section title="Error">
          <NativeText>{error}</NativeText>
        </Section>
      ) : null}

      <Section title="Requirement">
        <TextField ref={nameRef} placeholder="Math" onTextChange={setName} />
        <TextField ref={creditsRef} placeholder="4.0" onTextChange={setCredits} />
      </Section>

      {grouped.length ? (
        grouped.map((group) => (
          <Section key={group.year.id} title={group.year.name}>
            {group.rows.map((course) => {
              const isSelected = selected.includes(course.id);
              return (
                <NativeButton
                  key={course.id}
                  label={`${course.title ?? course.code ?? 'Course'}${isSelected ? '  ✓' : ''}`}
                  onPress={() => toggle(course.id)}
                />
              );
            })}
          </Section>
        ))
      ) : (
        <Section title="Courses">
          <NativeText>No courses to map yet.</NativeText>
        </Section>
      )}

      <Section>
        <NativeButton
          label={saving ? 'Saving…' : 'Save'}
          onPress={() => void save()}
          modifiers={[disabled(!valid || saving)]}
        />
      </Section>

      {id ? (
        <Section>
          <NativeButton
            label="Delete Requirement"
            role="destructive"
            systemImage="trash"
            onPress={remove}
          />
        </Section>
      ) : null}
    </NativeScreen>
  );
}
