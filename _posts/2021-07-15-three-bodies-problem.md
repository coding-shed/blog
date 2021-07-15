### Simulating trajectories of 3 celestial bodies
<p style='text-align: justify;'> Today we are going to simulate paths of 3 moving celestial bodies. To do so, we don't need maths or astro-physics degree. In today's day and age, world class computational tools are at out fingertips. We're going to leverage them to solve what hundred of years ago was deemed unsolvable.</p>

<br/><br/>
### Universal Law of Gravity
<p style='text-align: justify;'>
The following two equations, along with godlike ingenuity, allowed Issac Newton to determine orbital position of celestial bodies at any given point in time. First is the Universal Law of Gravity, which states that gravitational force between two bodies is proportional to their masses and inversely proportional to the square of their distance. </p>

\begin{aligned}
{F} = {G} \times \frac{m_1 m_2}{{r}^2}
\end{aligned}
<br/><br/>
<p style='text-align: justify;'>
Second equation is Newtown's Second Law of Motion. It tells us that with constant mass, net force applied to a body yields proportional acceleration.</p>

\begin{aligned}
{F} = {m_1} \times {a}
\end{aligned}
<br/><br/>
<p style='text-align: justify;'>
Combining those two facts i.e. equating those two definitions of force, gives a way to infer about object acceleration. But since we want to follow objects path in 3 dimensions, we have to turn above equation into vector form. Force and acceleration both have direction. In order to reflect that, we can add unit vector (with a hat) that points from body m<sub>1</sub> towards body m<sub>2</sub></p>

\begin{aligned}
\vec{a} = {G} \times \frac{m_2}{r^2} \hat{r}
\end{aligned}
<br/><br/>
<p style='text-align: justify;'>
And since unit vector is defined as vector divided by its norm, we can re-write the above into:</p>

\begin{aligned}
\vec{a} = {G} \times \frac{m_2}{r^3} \vec{r}_{12}
\end{aligned}
<br/><br/>
<p style='text-align: justify;'>
Subscript "12" means "from 1 towards 2" to describe vector's direction.</p>

<br/><br/>
### Working out the position
<p style='text-align: justify;'>
Now that we have acceleration, we can also derive body velocities and position. Acceleration is a derivative of velocity, and velocity is a derviative of position with respect to time. So, from acceleration we need to take two steps down: to velocity, and then to position. This can be expressed with two first order differential equations:</p>

\begin{aligned}
\frac{\partial \vec{v}_{i}}{\partial t} & = {G} \times \frac{m_j}{r^3_{ij}} \vec{r}_{ij} \\
\frac{\partial \vec{r}_{i}}{\partial t} & = \vec{v}_{i}
\end{aligned}
<br/><br/>


### Introducing third celestial body

<p style='text-align: justify;'>
Above equations are fine for systems with 2 bodies, not three. We still need to factor in the effect the third body has on both first and second body. With each body in question, the net exerted force will be the sum of forces from two remaining bodies. For example, acceleration of body 1 can be expressed as:</p>

\begin{aligned}
\frac{\partial \vec{v}_{1}}{\partial t} & = \frac{m_2}{r^3_{12}} \vec{r}_{12} + \frac{m_3}{r^3_{13}} \vec{r}_{13} \\
\end{aligned}
<br/><br/>

### Coding the equations
<p style='text-align: justify;'>
Now we are fully equiped to code a set of 3 acceleration equations. First, let's import all required packages. </p>


```python
import scipy as sci
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
import numpy as np
import pandas as pd
import scipy.integrate
```

<p style='text-align: justify;'>Function <b>equations_system</b> will accept state vector <b>w</b>, which comprises position coordinates of three bodies followed by velocity vectors, 18 floating point values in total: (3 bodies x 3 positions) + (3 bodies x 3 velocities) = 18. You can see that I skipped G constant in each acceleration equation. That's because we want all parameters provided to integrator to have similator magnitudes close to unity. This will make numerical methods work much better. One other solution would be to scale all of the parameters by some reference quantities to keep relative differences between them. But here, we are not mimicking any real celestial system. It's a toy example with completely arbitrary, random values.</p>


```python
def equations_system(w,t,m1,m2,m3):
    r1, r2, r3 = w[:3], w[3:6], w[6:9]
    v1, v2, v3 = w[9:12], w[12:15], w[15:18]

    r12=np.linalg.norm(r2-r1)
    r13=np.linalg.norm(r3-r1)
    r23=np.linalg.norm(r3-r2)
    
    a1=m2/r12**3*(r2-r1)+m3/r13**3*(r3-r1)
    a2=m1/r12**3*(r1-r2)+m3/r23**3*(r3-r2)
    a3=m1/r13**3*(r1-r3)+m2/r23**3*(r2-r3)

    r_derivs=np.concatenate((v1,v2,v3))
    v_derivs=np.concatenate((a1,a2,a3))
    derivs=np.concatenate((r_derivs,v_derivs))
    
    return derivs
```

<p style='text-align: justify;'>Next, we calculate magnitudes of distance vectors, remembering that vector AB from A to B is obtained by substracting initial point from terminal point: B - A. That's why r12 is defined as magnitude of r2-r1. Finally, we substitute state vector with updated values - what previosly was positions, now becomes velocity, and what was velocity is substituted with acceleration.</p>


<p style='text-align: justify;'>System of equation can now be fed into our main function that does all the work i.e. numerical integration along every time step provided. This is done by <b>odeint</b> function which requires: initial parameters (starting points), and amount of timesteps to perform calculations. Snapshot of positions at each timestep will create our trajectories.</p>


```python
def solve_equations(equations_system, initial_parameters, time_span, constants):
    solution=sci.integrate.odeint(equations_system,
                                        initial_parameters,
                                        time_span,
                                        args=constants)
    return solution
```

<p style='text-align: justify;'>Now we can initialise starting parameters i.e. individual masses of objects, positions and velocities along with timespan. Last two lines create dataframe from solution set and generate gif file visualizing the bodies trajectories. For clarity the visualization functions are put at the end of the blogpost. </p>


```python
m1=1.1
m2=0.907
m3=1.2

r1=np.random.uniform(low=-2, high=2, size=(3,))
r2=np.random.uniform(low=-2, high=2, size=(3,))
r3=np.random.uniform(low=-2, high=2, size=(3,))

v1=np.random.uniform(low=-0.5, high=0.5, size=(3,))
v2=np.random.uniform(low=-0.5, high=0.5, size=(3,))
v3=np.random.uniform(low=-0.5, high=0.5, size=(3,))

initial_parameters = np.array([r1,r2,r3,v1,v2,v3]).flatten()
time_span=np.linspace(0,30,700)
constants = (m1,m2,m3)

solution = solve_equations(equations_system, initial_parameters, time_span, constants)
df = build_dataframe(solution)
create_animation(f'animation_{num}.gif', df)
```

### Sample trajectories
<p style='text-align: justify;'>Below are sample trajectories with random initial parameters that I've found interesting. Feel free to play around with parameters - or even reproduce real-world 3 body systems!</p>

![](animation16.gif)

### Final remark
<p style='text-align: justify;'>There is a strong simplifying assumption made in the simulation. Not only we are assuming that gravitational center of mass is a infinitisemaly small point, but also that the entire object is this small point. This means we are assuming all bodies have volume equal 0. Under such assumption, the bodies will never collide. And from the animatrions above it is clear they should multiple times.</p>

![](animation9.gif)

![](animation54.gif)

![](animation58.gif)

### Code appendix

<p style='text-align: justify;'>

</p>


```python
def build_dataframe(solution):
    data = np.array([solution[:,:3], solution[:,3:6], solution[:,6:9]])
    df = pd.DataFrame({})
    for body in range(3):
        for dimension in range(3):
            df[f'body{body}_dim{dimension}'] = data[body][:,dimension]
    return df

def update_graph(num, data, graph1, graph2, graph3, dots1, dots2, dots3):
    graph1.set_data(df.body0_dim0[:num+1], df.body0_dim1[:num+1])
    graph1.set_3d_properties(df.body0_dim2[:num+1])
    
    graph2.set_data(df.body1_dim0[:num+1], df.body1_dim1[:num+1])
    graph2.set_3d_properties(df.body1_dim2[:num+1])
    
    graph3.set_data(df.body2_dim0[:num+1], df.body2_dim1[:num+1])
    graph3.set_3d_properties(df.body2_dim2[:num+1])
    
    dots1.set_data(df.body0_dim0[num], df.body0_dim1[num])
    dots1.set_3d_properties(df.body0_dim2[num])
    
    dots2.set_data(df.body1_dim0[num], df.body1_dim1[num])
    dots2.set_3d_properties(df.body1_dim2[num])
    
    dots3.set_data(df.body2_dim0[num], df.body2_dim1[num])
    dots3.set_3d_properties(df.body2_dim2[num])
    return graph1, graph2, graph3, dots1, dots2, dots3,

def define_ax(ax):
    ax.set_xlim3d(-2, 2)
    ax.set_ylim3d(-3, 1)
    ax.set_zlim3d(-3, 1)
    
    x, y, z = 10*(np.random.rand(3,1000)-0.5)
    ax.scatter(x, y, z, s=0.2, c='w')
    
    ax.set_facecolor('black') 
    ax.grid(False) 
    ax.w_xaxis.pane.fill = False
    ax.w_yaxis.pane.fill = False
    ax.w_zaxis.pane.fill = False

def create_animation(file_pathname, df):
    fig=plt.figure(figsize=(10,10))
    fig.set_facecolor('black')

    ax = Axes3D(fig)
    define_ax(ax)
    
    graph1, = ax.plot(df.body0_dim0, df.body0_dim1, df.body0_dim2, alpha=0.35, color="#FFFFF0")
    graph2, = ax.plot(df.body1_dim0, df.body1_dim1, df.body1_dim2, alpha=0.35, color="#FFFFCB")
    graph3, = ax.plot(df.body2_dim0, df.body2_dim1, df.body2_dim2, alpha=0.35, color="#F0FFFF")
    dots1, = ax.plot(df.body0_dim0[0], df.body0_dim1[0], df.body0_dim2[0], marker="o",linestyle='', 
                    markersize=12, color="#FFFFF0")
    dots2, = ax.plot(df.body1_dim0[0], df.body1_dim1[0], df.body1_dim2[0], marker="o",linestyle='', 
                    markersize=12, color="#FFFFCB")
    dots3, = ax.plot(df.body2_dim0[0], df.body2_dim1[0], df.body2_dim2[0], marker="o",linestyle='', 
                    markersize=12, color="#F0FFFF")

    ani = animation.FuncAnimation(fig, update_graph, 700, 
                                             fargs=(df, graph1, graph2, graph3, 
                                            dots1, dots2, dots3), 
                                            interval=500, blit=True)

    ani.save(file_pathname, writer='imagemagick', fps=24)

    #plt.show()
```


```python

```
