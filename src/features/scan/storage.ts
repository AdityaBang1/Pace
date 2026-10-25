/**
 * The files behind provenance.
 *
 * Rows and files are two stores, and the product makes promises about both: "Delete
 * all data" has to remove the photographs, and undo must not leave a page nobody can
 * reach. Nothing here touches a file outside the app's own `pages/` and `files/`
 * directories — a URI that still points into a picker cache is not ours to delete.
 */
import { Directory, File, Paths } from 'expo-file-system';

const OWNED_DIRECTORIES = ['pages', 'files'] as const;

function owned(uri: string): boolean {
  return OWNED_DIRECTORIES.some((name) => uri.includes(`/${name}/`));
}

/** Delete specific stored pages or PDFs; missing files are not an error. */
export async function deleteStoredSources(uris: (string | null | undefined)[]): Promise<void> {
  for (const uri of uris) {
    if (!uri || !owned(uri)) continue;
    try {
      const file = new File(uri);
      if (file.exists) file.delete();
    } catch {
      // A file that is already gone, or locked by the OS, must not fail an undo
      // whose database work has already committed.
    }
  }
}

/** Delete every page or PDF the app stored, then the (now empty) directories. */
export async function deleteAllStoredSources(uris: (string | null | undefined)[]): Promise<void> {
  await deleteStoredSources(uris);
  for (const name of OWNED_DIRECTORIES) {
    try {
      const directory = new Directory(Paths.document, name);
      if (directory.exists) directory.delete();
    } catch {
      // Same reasoning as above: the rows are gone, a stray file is not fatal.
    }
  }
}
