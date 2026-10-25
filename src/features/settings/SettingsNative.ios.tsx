/**
 * Settings, drawn by SwiftUI.
 *
 * A settings screen is a form: grouped sections, system rows, system switches. The
 * data and the actions come from `useSettings`, so this file is only the shape of
 * the thing, and the portable screen next to it stays behaviourally identical.
 */
import {
  Button as NativeButton,
  Form,
  Host,
  LabeledContent,
  Section,
  Text as NativeText,
  Toggle,
} from '@expo/ui/swift-ui';
import { useRouter } from 'expo-router';
import { View } from 'react-native';

import { PageHeader } from '../../ui/components';
import { useTheme } from '../../ui/useTheme';
import { useSettings } from './state';

export function SettingsNative() {
  const router = useRouter();
  const { palette } = useTheme();
  const state = useSettings();

  return (
    <View style={{ flex: 1, backgroundColor: palette.background }}>
      <PageHeader title="Settings" onBack={() => router.back()} />
      <Host style={{ flex: 1 }} useViewportSizeMeasurement>
        <Form>
          <Section title="Academic">
            <NativeButton
              label={`GPA Scale · ${state.gpaScale}`}
              systemImage="percent"
              onPress={() => router.push('/gpa-scale')}
            />
            <NativeButton label="Targets" systemImage="target" onPress={() => router.push('/targets')} />
            <NativeButton
              label="Requirements"
              systemImage="checklist"
              onPress={() => router.push('/requirements')}
            />
            <NativeButton label="Scores" systemImage="list.number" onPress={() => router.push('/tests')} />
            <NativeButton
              label="Transcript"
              systemImage="doc.text"
              onPress={() => router.push('/transcript')}
            />
          </Section>

          <Section title="Privacy" footer={<NativeText>Nothing leaves this device.</NativeText>}>
            <LabeledContent label="Remote AI">
              <NativeText>None</NativeText>
            </LabeledContent>
            <LabeledContent label="Uploads">
              <NativeText>None</NativeText>
            </LabeledContent>
            <LabeledContent label="Account">
              <NativeText>None</NativeText>
            </LabeledContent>
            <LabeledContent label="Documents stored">
              <NativeText>{String(state.documents)}</NativeText>
            </LabeledContent>
            <LabeledContent label="Courses stored">
              <NativeText>{String(state.courses)}</NativeText>
            </LabeledContent>
          </Section>

          <Section title="Reader">
            <LabeledContent label="Engine">
              <NativeText>{state.manifest?.engineVersion ?? '—'}</NativeText>
            </LabeledContent>
            <LabeledContent label="Record schema">
              <NativeText>{state.manifest?.schemaVersion ?? '—'}</NativeText>
            </LabeledContent>
            <LabeledContent label="Section model">
              <NativeText>{state.manifest?.models.section.id ?? '—'}</NativeText>
            </LabeledContent>
            <LabeledContent label="Runtime">
              <NativeText>On device</NativeText>
            </LabeledContent>
          </Section>

          <Section title="Storage" footer={<NativeText>The quoted source line is kept either way.</NativeText>}>
            <Toggle
              label="Keep page files"
              isOn={state.keepSource}
              onIsOnChange={state.setKeepSource}
            />
          </Section>

          <Section title="Data">
            <NativeButton
              label="Delete all data"
              systemImage="trash"
              role="destructive"
              onPress={state.confirmDelete}
            />
          </Section>

          {__DEV__ ? (
            <Section title="Development">
              <NativeButton
                label="Diagnostics"
                systemImage="stethoscope"
                onPress={() => router.push('/debug')}
              />
            </Section>
          ) : null}
        </Form>
      </Host>
    </View>
  );
}
