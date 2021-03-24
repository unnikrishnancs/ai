from Tkinter import * 
import tkMessageBox

#Event handler for click
def showMessage():
 tkMessageBox.showinfo(message="Name=%s, Department=%s, Salary=%s"%(tv.get(),de.get(),sa.get())) 

def reset():
 entName.delete(0,END)
 entDept.delete(0,END)
 entSal.delete(0,END)
 
 entName.focus()
 
mwin=Tk()

mwin.title("Employee Master")

mwin.geometry('300x300+70+70')

#Employee Name
lblName=Label(mwin,text="Name: ",font=("Arial"))
tv=StringVar()
entName=Entry(mwin,textvariable=tv)

#Employee Department
lblDept=Label(mwin,text="Dept: ",font=("Arial"))
de=StringVar()
entDept=Entry(mwin,textvariable=de)

#Employee Salary
lblSal=Label(mwin,text="Salary: ",font=("Arial"))
sa=StringVar()
entSal=Entry(mwin,textvariable=sa)

#Add buttons
btnDtls=Button(mwin,text="Show Details",command=showMessage)
btnCancel=Button(mwin,text="Reset",command=reset)

#Add widgets to main window
lblName.grid(column=0,row=0)
entName.grid(column=1,row=0)
lblDept.grid(column=0,row=1)
entDept.grid(column=1,row=1)
lblSal.grid(column=0,row=2)
entSal.grid(column=1,row=2)
btnDtls.grid(column=0,row=6)
btnCancel.grid(column=1,row=6)

entName.focus()

#Listen for events till the window is closed
mwin.mainloop()

