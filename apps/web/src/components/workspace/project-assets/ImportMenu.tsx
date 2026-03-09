import { useEffect, useRef, useState } from "react";

interface ImportMenuProps {
  onImportImages: () => void;
  onImportVideo: () => void;
  onImportWebcam: () => void;
}

export function ImportMenu({ onImportImages, onImportVideo, onImportWebcam }: ImportMenuProps) {
  const [open, setOpen] = useState(false);
  const menuRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    function handlePointerDown(event: MouseEvent) {
      if (!menuRef.current?.contains(event.target as Node)) setOpen(false);
    }
    window.addEventListener("mousedown", handlePointerDown);
    return () => window.removeEventListener("mousedown", handlePointerDown);
  }, []);

  function handleSelect(action: () => void) {
    setOpen(false);
    action();
  }

  return (
    <div className="import-menu" ref={menuRef}>
      <button type="button" className="primary-button import-menu-trigger" onClick={() => setOpen((value) => !value)} data-testid="asset-browser-import">
        Import
      </button>
      {open ? (
        <div className="import-menu-popover">
          <button type="button" className="ghost-button" onClick={() => handleSelect(onImportImages)}>
            Images
          </button>
          <button type="button" className="ghost-button" onClick={() => handleSelect(onImportVideo)}>
            Video File
          </button>
          <button type="button" className="ghost-button" onClick={() => handleSelect(onImportWebcam)}>
            Webcam Stream
          </button>
        </div>
      ) : null}
    </div>
  );
}
