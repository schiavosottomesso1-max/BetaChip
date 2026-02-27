import tkinter as tk
from tkinter import ttk, filedialog
from mmcensor.rt import mmc_realtime
import os
import sys
import importlib
import mmcensor.const as mmc_const
from functools import partial
import threading
import json

# ── Pink palette ────────────────────────────────────────────────────────────
_BG         = "#FFF0F5"   # lavender blush (window / frame background)
_ACCENT     = "#FF69B4"   # hot pink (buttons, highlights)
_ACCENT2    = "#FFB6C1"   # light pink (hover / secondary)
_ROSE_GOLD  = "#B76E79"   # rose-gold (shadow / border)
_TEXT_LIGHT = "white"
_TEXT_DARK  = "#4A0030"   # deep magenta for dark text on light bg
_LIST_BG    = "#FFE4EF"   # very pale pink for listbox / entry fields
_SEL_BG     = "#FF69B4"   # selection highlight
_REC_RED    = "#CC0000"   # red for active-recording indicator
_FONT_NORM  = ("Arial", 10)
_FONT_BOLD  = ("Arial", 10, "bold")

def _btn( parent, **kw ):
    """Helper: create a consistently styled pink Button."""
    defaults = dict( bg=_ACCENT, fg=_TEXT_LIGHT, relief="flat",
                     font=_FONT_BOLD, padx=10, pady=4, bd=0,
                     activebackground=_ACCENT2, activeforeground=_TEXT_DARK,
                     cursor="hand2" )
    defaults.update(kw)
    return tk.Button( parent, **defaults )

def _chk( parent, **kw ):
    """Helper: create a consistently styled pink Checkbutton."""
    defaults = dict( bg=_BG, fg=_TEXT_DARK, selectcolor=_ACCENT,
                     activebackground=_BG, font=_FONT_NORM )
    defaults.update(kw)
    return tk.Checkbutton( parent, **defaults )

def _lbl( parent, **kw ):
    """Helper: create a consistently styled pink Label."""
    defaults = dict( bg=_BG, fg=_TEXT_DARK, font=_FONT_NORM )
    defaults.update(kw)
    return tk.Label( parent, **defaults )

def _theme_frame( widget ):
    """Recursively apply the pink theme to an entire widget tree rooted at `widget`.
    Intended for frames built externally (e.g. decorator populate_config_frame calls)
    so their unstyled tk.Label / tk.Entry / tk.Checkbutton children all get the
    pink palette without touching the decorator logic files.
    """
    cls = widget.winfo_class()
    try:
        widget.configure( bg=_BG )
    except tk.TclError:
        pass
    if cls == "Entry":
        try:
            widget.configure( bg=_LIST_BG, fg=_TEXT_DARK, font=_FONT_NORM,
                               relief="flat", highlightthickness=1,
                               highlightcolor=_ACCENT, insertbackground=_TEXT_DARK )
        except tk.TclError:
            pass
    elif cls == "Checkbutton":
        try:
            widget.configure( fg=_TEXT_DARK, font=_FONT_NORM,
                               selectcolor=_ACCENT, activebackground=_BG )
        except tk.TclError:
            pass
    elif cls == "Label":
        try:
            widget.configure( fg=_TEXT_DARK, font=_FONT_NORM )
        except tk.TclError:
            pass
    for child in widget.winfo_children():
        _theme_frame( child )


class mmc_gui:
    
    def initialize( self ):
        ##########################
        ## make root window
        ## add save and load buttons
        ## make tabs
        ##########################
        self.root = tk.Tk()
        self.root.protocol( "WM_DELETE_WINDOW", self.on_close )
        self.root.geometry( "800x900" )
        self.root.configure( bg=_BG )
        self.root.minsize( 600, 600 )

        # ── Apply pink theme to ttk widgets ────────────────────────────────
        style = ttk.Style( self.root )
        style.theme_use("clam")
        style.configure( "TNotebook",       background=_BG,      borderwidth=0 )
        style.configure( "TNotebook.Tab",   background=_ACCENT2, foreground=_TEXT_DARK,
                          padding=[12, 4],  font=_FONT_BOLD )
        style.map( "TNotebook.Tab",
                   background=[("selected", _ACCENT), ("active", _ROSE_GOLD)],
                   foreground=[("selected", _TEXT_LIGHT)] )
        style.configure( "TFrame",          background=_BG )
        style.configure( "Pink.TCombobox",  fieldbackground=_LIST_BG,
                          background=_ACCENT2, foreground=_TEXT_DARK,
                          selectbackground=_SEL_BG, selectforeground=_TEXT_LIGHT )

        # ── Responsive root grid ─────────────────────────────────────────
        self.root.columnconfigure( 0, weight=1 )
        self.root.rowconfigure( 2, weight=1 )

        # Hero header: large "BETA CHIP" title with glow effect
        self.title_canvas = tk.Canvas( self.root, height=90, bg=_BG, highlightthickness=0 )
        self.title_canvas.grid( row=0, column=0, sticky="ew" )
        self.title_canvas.bind( "<Configure>", self._redraw_title )

        btn_bar = tk.Frame( self.root, bg=_BG )
        btn_bar.grid( row=1, column=0, sticky="ew", padx=4, pady=4 )

        self.save_button    = _btn( btn_bar, text="Save",                         command=self.save_pushed )
        self.save_as_button = _btn( btn_bar, text="Save As (not yet implemented)"                          )
        self.load_button    = _btn( btn_bar, text="Load",                         command=self.load_pushed )

        self.save_button.grid(    row=0, column=0, padx=4 )
        self.save_as_button.grid( row=0, column=1, padx=4 )
        self.load_button.grid(    row=0, column=2, padx=4 )

        tab_parent = ttk.Notebook( self.root )
        self.tab_decorate = ttk.Frame( tab_parent )
        self.tab_realtime = ttk.Frame( tab_parent )
        
        tab_parent.add( self.tab_decorate, text="Decorators" )
        tab_parent.add( self.tab_realtime, text="Realtime" )
        tab_parent.grid( row=2, column=0, sticky="nsew", padx=4, pady=4 )

        # ── Responsive tab grids ────────────────────────────────────────
        self.tab_realtime.columnconfigure( 0, weight=1 )
        self.tab_realtime.rowconfigure( 1, weight=1 )
        self.tab_decorate.columnconfigure( 0, weight=1 )
        self.tab_decorate.columnconfigure( 1, weight=1 )
        self.tab_decorate.rowconfigure( 1, weight=1 )

        #############################
        ## make realtime tab
        #############################
        self.rt = mmc_realtime()
        self.rt.initialize()
        self.rt.on_gray_callback = self.up
        self.rt.off_gray_callback = self.down

        rt_btn_bar = tk.Frame( self.tab_realtime, bg=_BG )
        rt_btn_bar.grid( row=0, column=0, sticky="ew", padx=2, pady=4 )

        self.ready_button      = _btn( rt_btn_bar, text="Make Ready",  command=self.make_ready_pushed )
        self.start_button      = _btn( rt_btn_bar, text="Start",       command=self.start_pushed,     state="disabled" )
        self.stop_button       = _btn( rt_btn_bar, text="Stop",        command=self.stop_pushed,      state="disabled" )
        self.screenshot_button = _btn( rt_btn_bar, text="Screenshot",  command=self.screenshot_pushed, state="disabled" )
        self.record_button     = _btn( rt_btn_bar, text="Record",      command=self.record_pushed,    state="disabled" )
        self.stop_record_button= _btn( rt_btn_bar, text="Stop Rec",    command=self.stop_record_pushed, state="disabled" )
        self.get_hwnds_button  = _btn( rt_btn_bar, text="Refresh Window List", command=self.refresh_hwnds )

        for col, btn in enumerate( [self.ready_button, self.start_button,
                                     self.stop_button, self.screenshot_button,
                                     self.record_button, self.stop_record_button,
                                     self.get_hwnds_button] ):
            btn.grid( row=0, column=col, padx=4 )

        # placeholder for realtime threads
        self.t_ready = None
        self.t_decorate = None

        self.known_hwnds = []

        # Listbox + scrollbar in a sub-frame
        list_frame = tk.Frame( self.tab_realtime, bg=_BG )
        list_frame.grid( row=1, column=0, sticky="nsew", padx=2, pady=2 )
        list_frame.columnconfigure( 0, weight=1 )
        list_frame.rowconfigure( 0, weight=1 )

        self.window_list = tk.Listbox(
            list_frame, selectmode="multiple", exportselection=0,
            bg=_LIST_BG, fg=_TEXT_DARK, selectbackground=_SEL_BG,
            selectforeground=_TEXT_LIGHT, font=_FONT_NORM,
            relief="flat", highlightthickness=1, highlightcolor=_ACCENT )
        _list_scroll = tk.Scrollbar( list_frame, orient="vertical",
                                     command=self.window_list.yview )
        self.window_list.configure( yscrollcommand=_list_scroll.set )
        self.window_list.bind( "<<ListboxSelect>>", self.change_hwnds_selection )
        self.window_list.grid( row=0, column=0, sticky="nsew" )
        _list_scroll.grid( row=0, column=1, sticky="ns" )

        self.refresh_hwnds()

        self.size_checks = []
        chk_frame = tk.Frame( self.tab_realtime, bg=_BG )
        chk_frame.grid( row=2, column=0, sticky="w", padx=4, pady=2 )
        for i in range( len( mmc_const.supported_sizes ) ):
            iv = tk.IntVar( value=(i < 2) )
            _chk( chk_frame,
                  text=f"net size {mmc_const.supported_sizes[i]}",
                  onvalue=1, offvalue=0, variable=iv,
                  command=self.update_sizes ).grid( row=0, column=i, padx=4 )
            self.size_checks.append( iv )

        ################################
        ## make decorator tab
        ################################
        self.decorator_widgets = []
        self.known_decorators = self.get_known_decorators()
        self.decorator_types = []
        self.selected_new_decorator = tk.StringVar()

        dec_ctrl = tk.Frame( self.tab_decorate, bg=_BG )
        dec_ctrl.grid( row=0, column=0, sticky="ew", padx=4, pady=4 )

        self.new_decorator_combobox = ttk.Combobox(
            dec_ctrl, textvariable=self.selected_new_decorator,
            values=self.known_decorators, style="Pink.TCombobox" )
        self.new_decorator_combobox.current(0)
        self.add_decorator_button = _btn( dec_ctrl, text="Add decorator",
                                          command=self.add_selected_decorator )

        self.new_decorator_combobox.grid( row=0, column=0, padx=4 )
        self.add_decorator_button.grid(   row=0, column=1, padx=4 )

        # Scrollable decorator list
        dec_outer = tk.Frame( self.tab_decorate, bg=_BG )
        dec_outer.grid( row=1, column=0, sticky="nsew", padx=4, pady=2 )
        dec_outer.columnconfigure( 0, weight=1 )
        dec_outer.rowconfigure( 0, weight=1 )

        dec_canvas = tk.Canvas( dec_outer, bg=_BG, highlightthickness=0 )
        dec_scroll = tk.Scrollbar( dec_outer, orient="vertical",
                                   command=dec_canvas.yview )
        dec_canvas.configure( yscrollcommand=dec_scroll.set )
        dec_canvas.grid( row=0, column=0, sticky="nsew" )
        dec_scroll.grid( row=0, column=1, sticky="ns" )

        self.decorators_frame = tk.Frame( dec_canvas, bg=_BG )
        self._dec_window = dec_canvas.create_window( (0, 0), window=self.decorators_frame, anchor="nw" )
        self.decorators_frame.bind(
            "<Configure>",
            lambda e: dec_canvas.configure(
                scrollregion=dec_canvas.bbox("all") ) )
        dec_canvas.bind(
            "<Configure>",
            lambda e: dec_canvas.itemconfigure( self._dec_window, width=e.width ) )

        self.decorator_config_frame = None
        self.decorator_being_configured = None

        self.load_pushed()
        self.update_sizes()

        self.root.mainloop()

    # ── Responsive title redraw ──────────────────────────────────────────
    def _redraw_title( self, event ):
        self.title_canvas.delete("all")
        cx = event.width // 2
        cy = 48
        # Draw glow/shadow layers in rose-gold
        for dx, dy in [(-2,-2),(2,-2),(-2,2),(2,2),(0,-3),(0,3),(-3,0),(3,0)]:
            self.title_canvas.create_text(
                cx+dx, cy+dy, text="✨ BETA CHIP ✨",
                font=("Arial", 46, "bold"), fill=_ROSE_GOLD, anchor="center" )
        # Draw main title in hot pink
        self.title_canvas.create_text(
            cx, cy, text="✨ BETA CHIP ✨",
            font=("Arial", 46, "bold"), fill=_ACCENT, anchor="center" )

    def up( self ):
        self.root.attributes( '-topmost', True )

    def down( self ):
        self.root.attributes( '-topmost', False )

    def update_sizes( self ):
        sizes = []
        for i in range(len(mmc_const.supported_sizes)):
            if self.size_checks[i].get():
                sizes.append( mmc_const.supported_sizes[i] )

        self.rt.update_sizes(sizes)

    def get_known_decorators( self ):
        paths = [ f.name for f in os.scandir('mmcensor/decorate') if f.is_dir() ]
        if '__pycache__' in paths:
            paths.remove( '__pycache__' )
        return( paths )

    def add_selected_decorator( self ):
        if len(self.selected_new_decorator.get() ):
            self.add_decorator( self.selected_new_decorator.get() )

    def add_decorator( self, decorator_type ):
        decorator = importlib.import_module( 'mmcensor.decorate.%s'%decorator_type ).decorator()
        decorator.initialize( mmc_const.nudenet_v3_classes )
        self.rt.decorators.append( decorator )
        self.decorator_types.append( decorator_type )
        self.redraw_decorators()

    def redraw_decorators( self ):
        for w in self.decorators_frame.winfo_children():
            w.destroy()

        self.decorator_being_configured = None

        for i in range(len(self.rt.decorators)):
            _btn( self.decorators_frame, text="x", padx=4,
                  command=partial( self.delete_decorator, i ) ).grid( row=i, column=0, padx=2, pady=2 )
            _lbl( self.decorators_frame, text=self.decorator_types[i] ).grid( row=i, column=1, padx=4 )
            _lbl( self.decorators_frame, text=self.rt.decorators[i].short_desc() ).grid( row=i, column=2, padx=4 )
            _btn( self.decorators_frame, text="configure",
                  command=partial( self.configure_decorator, i ) ).grid( row=i, column=3, padx=2, pady=2 )
            
    def delete_decorator( self, index ):
        self.rt.decorators.pop(index)
        self.decorator_types.pop(index)
        self.redraw_decorators()

    def _destroy_config_panel( self ):
        """Destroy and nullify the currently open decorator config panel, if any."""
        if self.decorator_config_frame is not None:
            self.decorator_config_frame.destroy()
            self.decorator_config_frame = None

    def configure_decorator( self, index ):
        # Destroy any previously open config panel first
        self._destroy_config_panel()
        self.redraw_decorators()

        # Outer panel — column 1 of tab_decorate
        self.decorator_config_frame = tk.Frame(
            self.tab_decorate, bg=_BG,
            highlightbackground=_ROSE_GOLD, highlightthickness=1 )

        # Header
        _lbl( self.decorator_config_frame,
              text=f"Configure: {self.decorator_types[index]}",
              font=_FONT_BOLD ).grid( row=0, column=0, columnspan=6,
                                      sticky="w", padx=8, pady=(8, 4) )

        # Content sub-frame populated by the decorator
        config_content = tk.Frame( self.decorator_config_frame, bg=_BG )
        config_content.grid( row=1, column=0, columnspan=6,
                             sticky="nsew", padx=8, pady=4 )
        self.rt.decorators[index].populate_config_frame( config_content )
        _theme_frame( config_content )

        # Action buttons inside the panel
        btn_row = tk.Frame( self.decorator_config_frame, bg=_BG )
        btn_row.grid( row=2, column=0, columnspan=6, pady=8, sticky="ew" )
        _btn( btn_row, text="Apply",
              command=partial( self.apply_decorator_config, index ) ).grid( row=0, column=0, padx=4 )
        _btn( btn_row, text="Close",
              command=partial( self.close_decorator_config, index ) ).grid( row=0, column=1, padx=4 )

        self.decorator_config_frame.grid( row=1, column=1, sticky="nsew", padx=8, pady=4 )

    def apply_decorator_config( self, index ):
        self.rt.decorators[index].apply_config_from_config_frame()

    def close_decorator_config( self, index ):
        self.rt.decorators[index].destroy_config_frame()
        self._destroy_config_panel()
        self.redraw_decorators()
    
    def save_pushed( self ):
        save_data = []
        for i in range( len( self.rt.decorators ) ):
            save_data.append( [ self.decorator_types[i], self.rt.decorators[i].export_settings() ] )

        with open('saved_settings.json', 'w') as f:
            json.dump( save_data, f )

    def load_pushed( self ):
        if not os.path.isfile( 'saved_settings.json' ):
            return

        with open('saved_settings.json' ) as data_file:
            save_data = json.load( data_file )

        self.rt.decorators.clear()
        self.decorator_types.clear()

        for elt in save_data:
            self.add_decorator( elt[0] )
            self.rt.decorators[-1].import_settings( elt[1] )

        self.redraw_decorators()

    def make_ready_pushed( self ):
        self.ready_button.config(state='disabled')
        self.t_ready = threading.Thread( target=self.make_ready_async )
        self.t_ready.daemon = True
        self.t_ready.start()

    def make_ready_async( self ):
        self.rt.make_ready()
        self.start_button.config(state='normal')

    def start_pushed( self ):
        self.start_button.config(state='disabled')
        self.t_decorate = threading.Thread( target=self.start_async )
        self.t_decorate.daemon = True
        self.t_decorate.start()
        self.screenshot_button.config(state='normal')
        self.stop_button.config(state='normal')
        self.record_button.config(state='normal')

    def start_async( self ):
        self.rt.go_decorate()
        self.start_button.config(state='normal')
        self.screenshot_button.config(state='disabled')
        self.stop_button.config(state='disabled')
        self.record_button.config(state='disabled')
        self.stop_record_button.config(state='disabled')

    def screenshot_pushed( self ):
        self.rt.take_screenshot()

    def record_pushed( self ):
        path = filedialog.asksaveasfilename(
            title="Save recording as…",
            defaultextension=".mp4",
            filetypes=[("MP4 video", "*.mp4"), ("AVI video", "*.avi"), ("All files", "*.*")] )
        if not path:
            return
        self.rt.start_recording( path )
        self.record_button.config(state='disabled')
        self.stop_record_button.config(state='normal', text='● Stop Rec', bg=_REC_RED)

    def stop_record_pushed( self ):
        self.rt.stop_recording()
        self.stop_record_button.config(state='disabled', text='Stop Rec', bg=_ACCENT)
        self.record_button.config(state='normal')

    def stop_pushed( self ):
        self.rt.running = False

    def refresh_hwnds( self ):
        print( 'refresh triggered' )
        self.known_hwnds = self.rt.sc.get_hwnds() # [ [ hwnd, description ] ]
        self.window_list.delete( 0, tk.END )
        for i in range(len(self.known_hwnds)):
            self.window_list.insert( tk.END, self.known_hwnds[i][1] )
            if self.known_hwnds[i][0] in self.rt.hwnds:
                self.window_list.selection_set( i )

        for hwnd in self.rt.hwnds:
            if hwnd not in (x[0] for x in self.known_hwnds):
                self.rt.hwnds.remove( hwnd )

    def change_hwnds_selection( self, evt ):
        print( 'change triggered' )
        chosen = self.window_list.curselection()
        self.rt.hwnds.clear()
        for i in chosen:
            self.rt.hwnds.append( self.known_hwnds[i][0] )

    def on_close( self ):
        self.rt.detector_async.shutdown()
        self.rt.running = False
        sys.exit()

