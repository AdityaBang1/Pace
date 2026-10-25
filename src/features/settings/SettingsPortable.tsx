/**
 * Settings on a platform without SwiftUI. Same rows, same actions, drawn with the
 * app's own primitives.
 */
import { useRouter } from 'expo-router';
import { Text, View } from 'react-native';

import { Button, Card, Divider, FieldRow, PageHeader, Screen, SectionHeader } from '../../ui/components';
import { NativeToggle } from '../../ui/NativeToggle';
import { space, type } from '../../ui/theme';
import { useTheme } from '../../ui/useTheme';
import { useSettings } from './state';

export function SettingsPortable() {
  const router = useRouter();
  const { palette } = useTheme();
  const state = useSettings();

  return (
    <Screen scroll>
      <PageHeader title="Settings" onBack={() => router.back()} />

      <SectionHeader>ACADEMIC</SectionHeader>
      <View style={{ paddingHorizontal: space.xl }}>
        <Card padded={false}>
          <View style={{ padding: space.lg }}>
            <FieldRow label="GPA Scale" value={state.gpaScale} onPress={() => router.push('/gpa-scale')} />
            <Divider />
            <FieldRow label="Targets" value="Set" onPress={() => router.push('/targets')} />
            <Divider />
            <FieldRow label="Requirements" value="Open" onPress={() => router.push('/requirements')} />
            <Divider />
            <FieldRow label="Scores" value="Open" onPress={() => router.push('/tests')} />
            <Divider />
            <FieldRow label="Transcript" value="Open" onPress={() => router.push('/transcript')} />
          </View>
        </Card>
      </View>

      <SectionHeader>PRIVACY</SectionHeader>
      <View style={{ paddingHorizontal: space.xl }}>
        <Card padded={false}>
          <View style={{ padding: space.lg }}>
            <FieldRow label="Remote AI" value="None" />
            <Divider />
            <FieldRow label="Uploads" value="None" />
            <Divider />
            <FieldRow label="Account" value="None" />
            <Divider />
            <FieldRow label="Documents stored" value={String(state.documents)} />
            <Divider />
            <FieldRow label="Courses stored" value={String(state.courses)} />
          </View>
        </Card>
      </View>

      <SectionHeader>READER</SectionHeader>
      <View style={{ paddingHorizontal: space.xl }}>
        <Card padded={false}>
          <View style={{ padding: space.lg }}>
            <FieldRow label="Engine" value={state.manifest?.engineVersion ?? '—'} />
            <Divider />
            <FieldRow label="Record schema" value={state.manifest?.schemaVersion ?? '—'} />
            <Divider />
            <FieldRow label="Section model" value={state.manifest?.models.section.id ?? '—'} />
          </View>
        </Card>
      </View>

      <SectionHeader>STORAGE</SectionHeader>
      <View style={{ paddingHorizontal: space.xl }}>
        <Card>
          <NativeToggle
            label="Keep page files"
            value={state.keepSource}
            onValueChange={state.setKeepSource}
          />
        </Card>
      </View>

      <SectionHeader>DATA</SectionHeader>
      <View style={{ paddingHorizontal: space.xl, gap: space.md }}>
        <Button title="Delete all data" kind="quiet" onPress={state.confirmDelete} />
      </View>

      {__DEV__ ? (
        <View style={{ padding: space.xl }}>
          <Button title="Diagnostics" kind="secondary" onPress={() => router.push('/debug')} />
        </View>
      ) : null}

      <View style={{ padding: space.xl }}>
        <Text style={[type.caption, { color: palette.inkFaint }]}>
          On device · Pace {state.version}
        </Text>
      </View>
    </Screen>
  );
}
